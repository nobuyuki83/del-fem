import numpy

def test_01():
    from del_msh_numpy import Polyline
    vtx2xyz_ini = Polyline.vtx2xyz_from_helix(30, 0.2, 0.2, 0.5)
    # print(vtx2xyz_ini.dtype)
    from del_fem.Rod3Darboux import Simulator
    simulator = Simulator(vtx2xyz_ini)
    simulator.vtx2isfix[0][:] = 1
    simulator.vtx2isfix[1][0:3] = 1
    simulator.vtx2isfix[-1][:] = 1
    simulator.vtx2isfix[-2][0:3] = 1
    simulator.initialize_with_perturbation(0.3, 0.1)
    assert (numpy.linalg.norm(simulator.vtx2framex_def, axis=1) - numpy.ones((simulator.vtx2framex_def.shape[0]),dtype=numpy.float32) ).max() < 1.0e-5
    # print(simulator.vtx2xyz_ini, simulator.vtx2xyz_def)
    for x in range(0,10):
        simulator.compute_rod_deformation_energy_grad_hessian()
        simulator.apply_fix_bc()
        print(simulator.w)
        simulator.update_solution_static()
    assert simulator.w < 1.0e-12

