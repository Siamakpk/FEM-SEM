# Installation

The scripts in this repository are not intended to be installed. The
dependencies are managed with [Pipenv] (see [the Pipenv website][Pipenv] for
installation instructions). To install the dependencies in a virtual
environment, run

```sh
pipenv sync
```

# Generating results

Run

```sh
pipenv run python3 "code/point-to-point error/Foil_winding_Vexcited_coupled.py" coupled=False
```

and

```sh
pipenv run python3 "code/point-to-point error/Foil_winding_Vexcited_coupled.py" coupled=True
```

These commands generate the raw data files `results/FEM.pkl` and
`results/FEMSEM.pkl`, respectively.

To generate the plots `results/Az.png` and `results/Jz.png` run

```sh
pipenv run python3 "code/point-to-point error/plot Az.py"
```

and

```sh
pipenv run python3 "code/point-to-point error/plot Jz.py"
```

[Pipenv]: https://pipenv.pypa.io
