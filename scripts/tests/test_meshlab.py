"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Fri Dec 01 2023
*  File : test_meshlab.py
******************************************* -->

"""

import pymeshlab


def example_load_mesh():
    # lines needed to run this specific example
    print('\n')
    # from . import samples_common
    # base_path = samples_common.samples_absolute_path()

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load a new mesh in the MeshSet, and sets it as current mesh
    # the path of the mesh can be absolute or relative
    ms.load_new_mesh("/home/levin/workspace/nerf/RoMe/outputs/wandb/run-20231129_153754-s8s24ga1/files/bev_mesh_epoch_7.obj")

    print(len(ms))  # now ms contains 1 mesh
    # instead of len(ms) you can also use:
    print(ms.number_meshes())

    # set the first mesh (id 0) as current mesh
    ms.set_current_mesh(0)

    # print the number of vertices of the current mesh
    print(ms.current_mesh().vertex_number())

    out_dict = ms.get_geometric_measures()


    output_path = '/home/levin/workspace/nerf/RoMe/outputs/wandb/run-20231129_153754-s8s24ga1/files/'
    # applying some filters...
    ms.generate_simplified_point_cloud()
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=1024)
    ms.save_current_mesh(output_path + 'chameleon_simplified.obj')
    ms.compute_texmap_from_color(textname='chameleon_simplified.png')

    # get a reference to the current mesh
    m = ms.current_mesh()

    # get numpy arrays of vertices and faces of the current mesh
    v_matrix = m.vertex_matrix()
    f_matrix = m.face_matrix()

    # clear the MeshSet
    ms.clear()

    # create a mesh cube into the MeshSet
    ms.create_cube()

    # compute an edge mesh composed of a planar section of the mesh
    # default values will use a plane with +X normal and passing into the origin
    # a new mesh will be added into the MeshSet and will be the current one
    ms.generate_polyline_from_planar_section()

    # get a reference to the current edge mesh
    m = ms.current_mesh()

    # get numpy arrays of vertices, edges and faces of the current mesh
    v_matrix = m.vertex_matrix()
    e_matrix = m.edge_matrix()
    f_matrix = m.face_matrix()
    return




example_load_mesh()
