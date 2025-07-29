import numpy

class Simulator:
    def __init__(self, vtx2xyz_ini):
        num_vtx = vtx2xyz_ini.shape[0]
        assert vtx2xyz_ini.shape == (num_vtx,3)
        self.vtx2xyz_ini = vtx2xyz_ini.copy()
        from del_msh_numpy import Polyline
        self.vtx2framex_ini = Polyline.vtx2framex_from_vtx2xyz(vtx2xyz_ini)
        self.vtx2xyz_def = self.vtx2xyz_ini.copy()
        self.vtx2framex_def = self.vtx2framex_ini.copy()
        self.vtx2isfix = numpy.ndarray(shape=(num_vtx,4), dtype=numpy.int32)
        self.w = numpy.array(0., dtype=numpy.float32)
        self.dw = numpy.ndarray(shape=(num_vtx,4), dtype=numpy.float32)
        (self.row2idx, self.idx2col) = Polyline.vtx2vtx_rods(numpy.array([0, num_vtx], dtype=numpy.uint64))
        self.row2val = numpy.ndarray(shape=(num_vtx,16), dtype=numpy.float32)
        num_idx = self.idx2col.shape[0]
        self.idx2val = numpy.ndarray(shape=(num_idx,16), dtype=numpy.float32)
        # print(self.row2idx)
        # print(self.idx2col)
        # del_msh_numpy.Polyline.
        # from del_msh_numpy.Polyline polyline_vtx2vtx_rods
        self.u_vec = numpy.ndarray(shape=(num_vtx,4), dtype=numpy.float32)
        self.p_vec = numpy.ndarray(shape=(num_vtx,4), dtype=numpy.float32)
        self.ap_vec = numpy.ndarray(shape=(num_vtx,4), dtype=numpy.float32)

    def initialize_with_perturbation(self, pos_mag, framex_mag):
         from .del_fem import rod3_darboux_initialize_with_perturbation
         rod3_darboux_initialize_with_perturbation(
             self.vtx2xyz_def,
             self.vtx2framex_def,
             self.vtx2xyz_ini,
             self.vtx2framex_ini,
             self.vtx2isfix,
             pos_mag,
             framex_mag)

    def compute_rod_deformation_energy_grad_hessian(self):
        self.row2val.fill(0.)
        self.idx2val.fill(0.)
        from .del_fem import add_wdwddw_rod3_darboux
        add_wdwddw_rod3_darboux(
            self.vtx2xyz_ini,
            self.vtx2framex_ini,
            self.vtx2xyz_def,
            self.vtx2framex_def,
            self.w,
            self.dw,
            self.row2idx,
            self.idx2col,
            self.row2val,
            self.idx2val)

    def apply_fix_bc(self):
        from .del_fem import block_sparse_apply_bc
        block_sparse_apply_bc(
            1.0,
            self.vtx2isfix,
            self.row2val,
            self.idx2val,
            self.row2idx,
            self.idx2col)
        from .del_fem import block_sparse_set_fixed_bc_to_rhs_vector
        block_sparse_set_fixed_bc_to_rhs_vector(
            self.vtx2isfix,
            self.dw)


    def update_solution_static(self):
        from .del_fem import conjugate_gradient
        conv = conjugate_gradient(
            self.dw,
            self.u_vec,
            self.p_vec,
            self.ap_vec,
            self.row2idx,
            self.idx2col,
            self.idx2val,
            self.row2val)
        # print(conv)
        from .del_fem import rod3_darboux_update_solution_hair
        rod3_darboux_update_solution_hair(
           self.vtx2xyz_def,
           self.vtx2framex_def,
           self.u_vec,
           self.vtx2isfix)


