import gmsh
import numpy

def generate_geometry(file,
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
                        nelems_window_boundary):


    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("{}".format(file))


    # Winding window width calculation
    window_w = clear_core_LV + m_LV * copper_w \
               + (m_LV - 1) * ins_w + clear_LV_HV + m_HV * copper_w + (m_HV - 1) * ins_w + clear_core_HV

    # Winding window height calculation
    window_h = 2 * clear_h + copper_h

    # Add geometry points
    # Window 
    gmsh.model.geo.addPoint(0, 0, 0, tag=1)                                     #core outer 1
    gmsh.model.geo.addPoint(window_w, 0, 0,  tag=2)                    #core outer 2
    gmsh.model.geo.addPoint(0, window_h, 0, tag=3)                     #core outer 3
    gmsh.model.geo.addPoint(window_w, window_h, 0, tag=4)     #core outer 4
    
    # Coils
    #bottom points
    point_tag_coil = 101
    np = 1
    while np <= m_LV:
        #bottom points for left side coils
        gmsh.model.geo.addPoint(clear_core_LV + (np-1) * copper_w + (np-1) * ins_w, clear_h, 0,  tag=point_tag_coil)
        point_tag_coil += 1
        gmsh.model.geo.addPoint(clear_core_LV + (np) * copper_w + (np-1) * ins_w, clear_h, 0, tag=point_tag_coil)
        point_tag_coil += 1
        np += 1

    np = 1
    while np <= m_HV:
        # bottom points for right side coils
        gmsh.model.geo.addPoint(clear_core_LV + (m_LV+np-1) * copper_w + (m_LV+np-2) * ins_w+ clear_LV_HV, clear_h, 0, tag=point_tag_coil)
        point_tag_coil += 1
        gmsh.model.geo.addPoint(clear_core_LV + (m_LV+np) * copper_w + (m_LV+np-2) * ins_w+ clear_LV_HV, clear_h, 0, tag=point_tag_coil)
        point_tag_coil += 1
        np += 1

    #top points
    point_tag_coil = 201
    np = 1
    while np <= m_LV:
        #top points for left side coils
        gmsh.model.geo.addPoint(clear_core_LV + (np-1) * copper_w + (np-1) * ins_w, clear_h + copper_h, 0,  tag=point_tag_coil)
        point_tag_coil += 1
        gmsh.model.geo.addPoint(clear_core_LV + (np) * copper_w + (np-1) * ins_w, clear_h + copper_h, 0, tag=point_tag_coil)
        point_tag_coil += 1
        np += 1


    np = 1
    while np <= m_HV:
        # top points for right side coils
        gmsh.model.geo.addPoint(clear_core_LV + (m_LV+np-1) * copper_w + (m_LV+np-2) * ins_w+ clear_LV_HV, clear_h + copper_h, 0, tag=point_tag_coil)
        point_tag_coil += 1
        gmsh.model.geo.addPoint(clear_core_LV + (m_LV+np) * copper_w + (m_LV+np-2) * ins_w+ clear_LV_HV, clear_h + copper_h, 0, tag=point_tag_coil)
        point_tag_coil += 1
        np += 1

    # Connect the points in lines
    # Window
    gmsh.model.geo.addLine(1, 2, 1001)
    gmsh.model.geo.addLine(3, 4, 1002)
    gmsh.model.geo.addLine(1, 3, 1003)
    gmsh.model.geo.addLine(2, 4, 1004)

    # coils
    # Bottom
    point_tag_line = 1101

    np = 1
    point = 101

    while np <= m_LV+m_HV:
        gmsh.model.geo.addLine(point, point+1, point_tag_line)

        point += 2
        point_tag_line += 1
        np += 1

    # top
    point_tag_line = 1201
    np = 1
    point = 201

    while np <= m_LV+m_HV:
        gmsh.model.geo.addLine(point, point+1, point_tag_line)

        point += 2
        point_tag_line += 1
        np += 1

    # vertical lines
    point_tag_line = 1301
    np = 1
    point = 101

    while np <= 2*(m_LV + m_HV):
        gmsh.model.geo.addLine(point, point + 100, point_tag_line)

        point += 1
        point_tag_line += 1
        np += 1



    # Group the outside lines into a loop
    # Window
    gmsh.model.geo.addCurveLoop([1001, -1002, -1003, 1004], 2000)



    # Coils curve tag
    curve_tag = 2002
    np = 1
    line_tag_horizontal = 1101
    line_tag_vertical = 1301
    while np <= (m_LV + m_HV):

        gmsh.model.geo.addCurveLoop([line_tag_vertical, -line_tag_vertical-1, -line_tag_horizontal, line_tag_horizontal+100], curve_tag)
        curve_tag += 1
        line_tag_vertical += 2
        line_tag_horizontal += 1
        np += 1

    # Make a surface of the loop
    
    # Window
    plane_tag = 3000
    Window_plane = -1 * numpy.arange(2002,2002+m_HV+m_LV)
    Window_plane = numpy.append(2000,Window_plane)
    gmsh.model.geo.addPlaneSurface(Window_plane, plane_tag)


    
    np = 1
    plane_tag += 2

    curve_tag = 2002
    while np <= m_LV+m_HV :
        gmsh.model.geo.addPlaneSurface([curve_tag], plane_tag)
        plane_tag += 1
        curve_tag += 1
        np += 1



    # Group the model entities
    # The lines that make up the boundary box
    gmsh.model.addPhysicalGroup(1, [1001, 1002, 1003, 1004], 4100)
    gmsh.model.addPhysicalGroup(1, [1001], 4101)
    gmsh.model.addPhysicalGroup(1, [1002], 4102)
    gmsh.model.addPhysicalGroup(1, [1003], 4103)
    gmsh.model.addPhysicalGroup(1, [1004], 4104)


    # Air
    gmsh.model.addPhysicalGroup(2, [3000], 4000)    
    

    
    # # coils left
    n = 1
    for x in range(3002, 3002+m_LV, 1):
        gmsh.model.addPhysicalGroup(2, [int(x)], n)
        gmsh.model.setPhysicalName(2, n, f'cond{n}')
        n +=1
    
    for x in range(3002+m_LV, 3002+m_HV+m_LV, 1):
        gmsh.model.addPhysicalGroup(2, [int(x)], n)
        gmsh.model.setPhysicalName(2, n, f'cond{n}')
        n +=1
    # # Coils right
    # gmsh.model.addPhysicalGroup(2, [int(x) for x in range(3002+m_LV, 3002+m_HV+m_LV+1, 1)], 4003)




    # # Set the names for the groups
    #
    gmsh.model.setPhysicalName(2, 4000, "air")



    gmsh.model.setPhysicalName(1, 4101, "top")
    gmsh.model.setPhysicalName(1, 4102, "bottom")
    gmsh.model.setPhysicalName(1, 4103, "left")
    gmsh.model.setPhysicalName(1, 4104, "right")

    gmsh.model.setPhysicalName(1, 4100, "boundary")


    # We then generate a 2D mesh...


    np = 1
    line_tag_horizontal = 1101
    line_tag_vertical = 1301

    while np <= 2*(m_LV + m_HV):

        gmsh.model.geo.mesh.setTransfiniteCurve(line_tag_vertical, nelems_copper_h)
        line_tag_vertical += 1
        np += 1

    np = 1
    while np <= (m_LV + m_HV):
        gmsh.model.geo.mesh.setTransfiniteCurve(line_tag_horizontal, nelems_copper_w)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_tag_horizontal+100, nelems_copper_w)

        line_tag_horizontal += 1
        np += 1


    np = 1
    while np <= (m_LV + m_HV+1):
        gmsh.model.geo.mesh.setTransfiniteSurface(3001+np)
        np+=1

    gmsh.model.geo.mesh.setTransfiniteCurve(1001, nelems_window_boundary)
    gmsh.model.geo.mesh.setTransfiniteCurve(1002, nelems_window_boundary)
    gmsh.model.geo.mesh.setTransfiniteCurve(1003, nelems_window_boundary)
    gmsh.model.geo.mesh.setTransfiniteCurve(1004, nelems_window_boundary)


    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.recombine()
    # gmsh.model.mesh.setRecombine(2, 3001)




    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

    # and save it to disk
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write("{}.msh".format(file))

    # if '-nopopup' not in sys.argv:
    # gmsh.fltk.run()
    # This should be called at the end
    gmsh.finalize()

# generate_geometry()
