from dataclasses import dataclass
import numpy, scipy.linalg, treelog, time, matplotlib, itertools, functools, operator
import nutils
from nutils import cli, evaluable, export, function, mesh, sample, solver, types
from nutils.expression_v2 import Namespace
from nutils.topology import Topology
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple, Union
import math
import os
import pickle 
from pathlib import Path
import sys


# Add the directory of this file the the modules search path such that we can
# import `Build_2D_geom_window_gmsh.py` from the same directory.
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.fspath(Path(__file__)))

def concat_verts_1d(*parts: Sequence[Iterable[float]]) -> Tuple[numpy.ndarray, List[int]]:
    ptr = [0]
    verts = list(parts[0])
    offset = verts.pop()
    for part in parts[1:]:
        ptr.append(len(verts))
        verts.extend(vert + offset for vert in part)
        offset = verts.pop()
    ptr.append(len(verts))
    verts.append(offset)
    return numpy.array(verts), ptr

def uniform_verts(width: float, *, nelems: Optional[int] = None, elem_size: Optional[int] = None) -> numpy.ndarray:
    if not (bool(elem_size) ^ bool(nelems)):
        raise ValueError('either `nelems` or `elem_size` must be specified')
    elif nelems is None:
        nelems = max(1, int(round(width / elem_size)))
    return numpy.linspace(0, width, nelems + 1)

def exponential_verts(width: float, *, left: Optional[float] = None, right: Optional[float] = None):
    if not right:
        if width < left:
            return uniform_verts(width, nelems=1)
        verts = [0]
        while verts[-1] < width:
            verts.append(verts[-1] + left)
            left *= 2
        if verts[-1] - width > width - verts[-2]:
            verts.pop()
        scale = width / verts[-1]
        return numpy.fromiter((scale * vert for vert in verts), dtype=float)
    elif not left:
        return width - exponential_verts(width, left=right)[::-1]
    elif left and right:
        return concat_verts_1d(exponential_verts(width / 2, left=left), exponential_verts(width / 2, right=right))[0]
    else:
        raise ValueError('either `left` or `right` must be specified')

def join_bases(base_topo, sub_topos_bases):
    zcoeffs = numpy.zeros((0,1))
    zdofs = numpy.zeros((0,), dtype=int)
    coeffs = list(zcoeffs for ielem in range(len(base_topo)))
    dofs = list(zdofs for ielem in range(len(base_topo)))
    offset = 0
    for sub_topo, sub_basis in sub_topos_bases:
        for isub_elem, trans in enumerate(sub_topo.transforms):
            ibase_elem = base_topo.transforms.index(trans)
            if coeffs[ibase_elem].shape[0]:
                raise ValueError('overlapping subtopologies')
            coeffs[ibase_elem] = sub_basis.get_coefficients(isub_elem)
            dofs[ibase_elem] = sub_basis.get_dofs(isub_elem) + offset
        offset += sub_basis.ndofs
    return function.PlainBasis(coeffs, dofs, offset, base_topo.f_index, base_topo.f_coords)

class ConditionalCosineBasis(function.Array):

    def __init__(self, k, x, condition):
        self.k = numpy.array(k)
        self.x = function.Array.cast(x)
        self.condition = function.Array.cast(condition, dtype=int)
        assert self.k.ndim == 1
        assert self.x.ndim == 0
        assert self.condition.ndim == 0
        super().__init__(
            shape=k.shape,
            dtype=float,
            spaces=self.x.spaces | self.condition.spaces,
            arguments=self.x.arguments | self.condition.arguments,
        )

    def lower(self, args):
        condition = self.condition.lower(function.LowerArgs((), args.transform_chains, {}))
        dofs = evaluable.Range(evaluable.Take(evaluable.asarray([0, self.shape[0]]), condition))
        k = evaluable.Take(evaluable.asarray(self.k), dofs)
        x = self.x.lower(args)
        f = evaluable.cos(evaluable.prependaxes(k, x.shape) * evaluable.appendaxes(x, k.shape))
        return evaluable.Inflate(f, dofs, evaluable.asarray(self.shape[0]))

@dataclass
class VoltageSource:

    voltage: float

    def constrain(self, V, I):
        return V - self.voltage

@dataclass
class CurrentSource:

    current: float

    def constrain(self, V, I):
        return I - self.current

@dataclass
class ShortCircuit:

    def constrain(self, V, I):
        return V

@dataclass
class OpenCircuit:

    def constrain(self, V, I):
        return I

def main(
        # problem parameters
        freq: float = 20e3,
        conductivity: float = 5.998e7,     # Copper conductivity
        source_LV: Union[CurrentSource, VoltageSource, ShortCircuit, OpenCircuit] = CurrentSource(1),
        source_HV: Union[CurrentSource, VoltageSource, ShortCircuit, OpenCircuit] = ShortCircuit(),
        # domain parameters
        m_LV: int = 10,                    # No. of turns left coil
        m_HV: int = 10,                    # No. of turns right coil
        clear_core_LV: float =20e-3,       # isolation distance left
        clear_core_HV: float = 20e-3,      # isolation distance right
        clear_LV_HV: float = 20e-3,        # isolation distance inter-winding
        clear_h: float=20e-3,              # isolation distance up/down
        copper_w: float = 1e-3,              # foil thickness
        copper_h: float = 100e-3,              # foil height
        ins_w: float = 0.2e-3,             # inter-layer insulation thickness
        # discretisation parameters
        meshtype: str = 'rectilinear',
        nelems_copper_w: int = 1,          # No. of mesh elements
        nelems_copper_h: int = 30,        # No. of mesh elements
        nelems_window_boundary: int = 25,
        btype: str = 'std',
        degree: int = 2,
        coupled: bool = True,
        # postprocessing parameters
        plot_hull: bool = True,
        spy_plot: bool = False,
        nfuncs_sem_x: int = 6,
        nfuncs_sem_y: int = 30,
        results_dir: Path = Path('results')):

    ns = Namespace()
    ns.π = numpy.pi
    ns.μ0 = '4e-7 π'
    ns.j = 1j
    ns.freq = freq
    ns.ω = '2 π freq'
    degree2 = degree*2

    if meshtype == 'gmsh':

        import Build_2D_geom_window_gmsh
        Build_2D_geom_window_gmsh.generate_geometry(os.fspath(results_dir / "foil_MFT"),
                                m_LV,  #No. of turns left coil
                                m_HV, #No. of turns right coil
                                clear_core_LV,   #isolation distance left
                                clear_core_HV,   #isolation distance right
                                clear_LV_HV,    #isolation distance inter-winding
                                clear_h,        #isolation distance up/down
                                copper_w,      #foil thickness
                                copper_h,       #foil height
                                ins_w,         #inter-layer insulation thickness
                                nelems_copper_w,    #No. of mesh elements
                                nelems_copper_h,   #No. of mesh elements
                                nelems_window_boundary ) #
        X , ns.x = nutils.mesh.gmsh(os.fspath(results_dir / 'foil_MFT.msh')) # Import the Gmsh file and construct the namespace

    elif meshtype == 'rectilinear':

        # Rectilinear mesh with elements that double in size away from edges.
        # The element size at the edges is controlled entirely by
        # `nelems_copper_w`.
        elem_size_edge = copper_w / nelems_copper_w

        copper_verts = exponential_verts(copper_w, left=elem_size_edge, right=elem_size_edge)
        ins_verts = exponential_verts(ins_w, left=elem_size_edge, right=elem_size_edge)
        parts = [exponential_verts(clear_core_LV, right=elem_size_edge), copper_verts]
        for i in range(1, m_LV):
            parts += [ins_verts, copper_verts]
        parts += [exponential_verts(clear_LV_HV, left=elem_size_edge, right=elem_size_edge), copper_verts]
        for i in range(1, m_HV):
            parts += [ins_verts, copper_verts]
        parts += [exponential_verts(clear_core_HV, left=elem_size_edge)]
        verts_x, ptr_x = concat_verts_1d(*parts)

        verts_y, ptr_y = concat_verts_1d(
            exponential_verts(clear_h, right=elem_size_edge),
            exponential_verts(copper_h, left=elem_size_edge, right=elem_size_edge),
            exponential_verts(clear_h, left=elem_size_edge),
        )

        subs_x = 'slice_air_left', 'slice_conds_LV', 'slice_air_center', 'slice_conds_HV', 'slice_air_right'
        sub_ptr_x = ptr_x[0], ptr_x[1], ptr_x[2*m_LV], ptr_x[2*m_LV+1], ptr_x[-2], ptr_x[-1]

        X, ns.x = mesh.rectilinear([verts_x, verts_y])
        X = X.withsubdomain(
            **{f'cond{i+1}': X[ptr_x[2*i+1]:ptr_x[2*i+2],ptr_y[1]:ptr_y[2]] for i in range(m_LV)},
            **{f'cond{i+1}': X[ptr_x[2*i+1]:ptr_x[2*i+2],ptr_y[1]:ptr_y[2]] for i in range(m_LV, m_LV+m_HV)},
            **{k: X[i:j] for k, (i, j) in zip(subs_x, itertools.pairwise(sub_ptr_x))},
        )
        X = X.withsubdomain(air=X - X[','.join(f'cond{i+1}' for i in range(m_LV + m_HV))])

        ifaces = [X[f'slice_conds_{k}'].boundary[side] for k in ('LV', 'HV') for side in ('left', 'right')]

    else:

        raise ValueError(f'unknown mesh type: {meshtype}')

    X = X.withsubdomain(
        conds_LV=','.join(f'cond{k+1}' for k in range(m_LV)),
        conds_HV=','.join(f'cond{k+1+m_LV}' for k in range(m_HV)),
        all_conds=','.join(f'cond{k+1}' for k in range(m_LV + m_HV)),
    )

    ns.define_for('x', gradient='∇', jacobians=('dV', 'dS'), normal='n')
    ns.D = function.levicivita(2)
    ns.t_i = 'D_ij n_j' # surface tangent

    ns.σ = conductivity * X.indicator('all_conds')
    ns.nu = '1 / μ0'

    if btype == 'spline':
        mk_fem_basis_xy = lambda subtopo, i0, i1: subtopo.basis(btype, degree=degree, knotvalues=[verts_x[i0:(len(verts_x)+i1 if i1 < 0 else i1)+1], verts_y])
        mk_fem_basis_y = lambda iface: iface.basis(btype, degree=degree, knotvalues=[verts_y])
    else:
        mk_fem_basis_xy = lambda subtopo, i0, i1: subtopo.basis(btype, degree=degree)
        mk_fem_basis_y = lambda iface: iface.basis(btype, degree=degree)

    # Build a basis for `Az`.
    if coupled:

        if meshtype != 'rectilinear':
            raise NotImplementedError(f'the combination `meshtype={meshtype}` and `coupled=True` is not implemented')

        bases = []
        sem_parts = set()

        # SEM bases

        k_y = numpy.arange(nfuncs_sem_y)
        arg_y = numpy.pi * (ns.x[1] - verts_y[0]) / (verts_y[-1] - verts_y[0])
        basis_y = ConditionalCosineBasis(k_y, arg_y, X.indicator('slice_air_left,slice_air_center,slice_air_right'))

        x0, x1 = verts_x[sub_ptr_x[0]], verts_x[sub_ptr_x[1]]
        k_x = numpy.concatenate([[0], 2 * numpy.arange(nfuncs_sem_x - 1) + 1])
        basis_x = ConditionalCosineBasis(k_x, numpy.pi * (ns.x[0] - x0) / (2 * (x1 - x0)), X.indicator('slice_air_left'))
        bases.append(function.ravel(basis_x[:,numpy.newaxis] * basis_y[numpy.newaxis,:], 0))
        sem_parts.add('slice_air_left')

        x0 = verts_x[sub_ptr_x[-2]]
        x1 = verts_x[sub_ptr_x[-1]]
        k_x = numpy.concatenate([[0], 2 * numpy.arange(nfuncs_sem_x - 1) + 1])
        basis_x = ConditionalCosineBasis(k_x, numpy.pi * (x1 - ns.x[0]) / (2 * (x1 - x0)), X.indicator('slice_air_right'))
        bases.append(function.ravel(basis_x[:,numpy.newaxis] * basis_y[numpy.newaxis,:], 0))
        sem_parts.add('slice_air_right')

        x0 = verts_x[sub_ptr_x[2]]
        x1 = verts_x[sub_ptr_x[3]]
        k_x = numpy.concatenate([[0], 2 * numpy.arange(nfuncs_sem_x - 1) + 1])
        basis_x = ConditionalCosineBasis(k_x, numpy.pi * (ns.x[0] - x0) / (x1 - x0), X.indicator('slice_air_center'))
        bases.append(function.ravel(basis_x[:,numpy.newaxis] * basis_y[numpy.newaxis,:], 0))
        basis_x = ConditionalCosineBasis(k_x[1:], numpy.pi * (ns.x[0] - x0) / (x1 - x0) + numpy.pi / 2, X.indicator('slice_air_center'))
        bases.append(function.ravel(basis_x[:,numpy.newaxis] * basis_y[numpy.newaxis,:], 0))
        sem_parts.add('slice_air_center')

        # FEM bases
        bases.append(join_bases(
            X,
            [
                (X[k], mk_fem_basis_xy(X[k], i0, i1))
                for k, (i0, i1) in zip(subs_x, itertools.pairwise(sub_ptr_x))
                if k not in sem_parts
            ],
        ))

        A_basis = numpy.concatenate(bases)

    else:

        A_basis = mk_fem_basis_xy(X, 0, -1)

    ns.add_field(('Az', 'β'), A_basis, dtype=complex)
    ns.add_field(('VtermL', 'VtermH', 'ItermL', 'ItermH'), dtype=complex)
    ns.B_i = 'D_ij ∇_j(Az)'
    ns.dzφ = function.dotarg('Vcond', numpy.stack([X.indicator(f'cond{i+1}') for i in range(m_LV+m_HV)]), dtype=complex)
    ns.Ez = 'dzφ - j ω Az '
    ns.Jz = 'σ Ez'

    cons = {}

    res = X.integral('(nu ∇_j(β) ∇_j(Az) - β Jz) dV' @ ns, degree=degree2)
    targets = ['Az:β']

    if coupled:
        for i, iface in enumerate(ifaces):
            res += iface.integral('[β] {∇_i(Az)} n_i dS' @ ns, degree=degree2)
            λ = function.dotarg(f'λjumpAz{i}', mk_fem_basis_y(iface), dtype=complex)
            weak_cons = iface.integral(λ * function.jump(ns.Az) * ns.dS, degree=degree2)
            res += function.linearize(weak_cons, f'Az:β,λjumpAz{i}:γjumpAz{i}')
            targets += [f'λjumpAz{i}:γjumpAz{i}']

    # CIRCUIT CONSTRAINTS
    #
    # We populate the list `circuit_cons` with all constraints, then we
    # multiply every constraint with a lagrange multiplier and add the
    # linearization to `res`.

    circuit_cons = []

    # All conductors in a circuit have the same current as the terminal,
    # `ns.Iterm?`.

    for icond in range(m_LV):
        circuit_cons.append(ns.ItermL - X[f'cond{icond+1}'].integral('Jz dV' @ ns, degree=degree2))
    for icond in range(m_LV, m_LV+m_HV):
        circuit_cons.append(ns.ItermH - X[f'cond{icond+1}'].integral('Jz dV' @ ns, degree=degree2))

    # The voltage drops of all conductors in a circuit add up to the terminal
    # voltage, `ns.Vterm?`.

    Vcond = function.Argument('Vcond', shape=(m_LV+m_HV,), dtype=complex)
    circuit_cons.append(ns.VtermL - numpy.sum(numpy.take(Vcond, range(m_LV))))
    circuit_cons.append(ns.VtermH - numpy.sum(numpy.take(Vcond, range(m_LV, m_LV+m_HV))))

    # Apply the external constraints to the terminals.

    circuit_cons.append(source_LV.constrain(ns.VtermL, ns.ItermL))
    circuit_cons.append(source_HV.constrain(ns.VtermH, ns.ItermH))

    # Finally, linearize and add to `res`.

    circuit_targets = ['VtermL:UtermL', 'VtermH:UtermH', 'ItermL:JtermL', 'ItermH:JtermH', 'Vcond:Ucond', 'λcircuit:γcircuit']
    res += function.linearize(function.dotarg('λcircuit', numpy.stack(circuit_cons), dtype=complex), ','.join(circuit_targets))
    targets += circuit_targets

    # SOLVE

    # Plot the structure of the linear system.
    if spy_plot:
        matrix = res.derivative('β').derivative('Az').eval().export('dense')
        with export.mplfigure('spy.png') as fig:
            ax = fig.add_subplot(1, 1, 1)
            ax.spy(abs(matrix) > 1e-10)

    args = solver.solve_linear(','.join(targets), res, constrain=cons)

    return(ns.x,ns.Az,X,args,coupled,freq)


if __name__ == '__main__':
    cli.run(main)
