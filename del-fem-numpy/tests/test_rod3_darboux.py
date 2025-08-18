import numpy

def test_01():
    from del_msh_numpy import Polyline
    vtx2xyz_ini = Polyline.vtx2xyz_from_helix(30, 0.2, 0.2, 0.5)
    # print(vtx2xyz_ini.dtype)
    from del_fem_numpy.Rod3Darboux import Simulator
    simulator = Simulator(vtx2xyz_ini)
    simulator.vtx2isfix[0][:] = 1
    simulator.vtx2isfix[1][0:3] = 1
    simulator.vtx2isfix[-1][:] = 1
    simulator.vtx2isfix[-2][0:3] = 1
    simulator.initialize_with_perturbation(0.3, 0.1)
    assert (numpy.linalg.norm(simulator.vtx2framex_def, axis=1) - numpy.ones((simulator.vtx2framex_def.shape[0]),dtype=numpy.float32) ).max() < 1.0e-5
    # print(simulator.vtx2xyz_ini, simulator.vtx2xyz_def)
    for x in range(0,10):
        simulator.solve_static(None)
        print(simulator.w)
    assert simulator.w < 1.0e-12


def test_02():
    from del_msh_numpy import Polyline
    vtx2xyz_ini = Polyline.vtx2xyz_from_helix(30, 0.2, 0.2, 0.5)
    i_vtx_pull = 15
    pos_goal = numpy.array([1., -1.0, 0.2])
    # print(vtx2xyz_ini.dtype)
    from del_fem_numpy.Rod3Darboux import Simulator
    simulator = Simulator(vtx2xyz_ini)
    simulator.vtx2isfix[0][:] = 1
    simulator.vtx2isfix[1][0:3] = 1
    simulator.vtx2isfix[-1][:] = 1
    simulator.vtx2isfix[-2][0:3] = 1
    assert (numpy.linalg.norm(simulator.vtx2framex_def, axis=1) - numpy.ones((simulator.vtx2framex_def.shape[0]),dtype=numpy.float32) ).max() < 1.0e-5
    for x in range(0,10):
        simulator.solve_static((i_vtx_pull, pos_goal))
        print(simulator.w)
    Polyline.save_wavefront_obj(simulator.vtx2xyz_def, "../target/darboux_rod.obj")


def test_03():
    from del_msh_numpy import Polyline
    vtx2xyz_ini = Polyline.vtx2xyz_from_helix(30, 0.2, 0.2, 0.5)
    i_vtx_pull = 15
    pos_goal = vtx2xyz_ini[i_vtx_pull]# + numpy.array([0., 0., 0.])
    print(vtx2xyz_ini[i_vtx_pull], pos_goal)
    # print(vtx2xyz_ini.dtype)
    from del_fem_numpy.Rod3Darboux import Simulator
    simulator = Simulator(vtx2xyz_ini)
    simulator.vtx2isfix[0][:] = 1
    simulator.vtx2isfix[1][0:3] = 1
    simulator.vtx2isfix[-1][:] = 1
    simulator.vtx2isfix[-2][0:3] = 1
    simulator.initialize_with_perturbation(0.1, 0.01)
    assert (numpy.linalg.norm(simulator.vtx2framex_def, axis=1) - numpy.ones((simulator.vtx2framex_def.shape[0]),dtype=numpy.float32) ).max() < 1.0e-5
    # print(simulator.vtx2xyz_ini, simulator.vtx2xyz_def)
    print("################################")
    for x in range(0,100):
        simulator.solve_dynamic(1.0, (i_vtx_pull, pos_goal)) # pull nothing if the second argument is None
        print(simulator.w)
        # print(simulator.dw)
    Polyline.save_wavefront_obj(simulator.vtx2xyz_def, "../target/darboux_rod.obj")



