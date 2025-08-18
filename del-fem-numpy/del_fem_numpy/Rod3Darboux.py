import numpy
import numpy.typing as npt
from typing import Optional, Tuple

class Simulator:
    def __init__(self, vtx2xyz_ini):
        num_vtx = vtx2xyz_ini.shape[0]
        assert vtx2xyz_ini.shape == (num_vtx,3)
        self.vtx2xyz_ini = vtx2xyz_ini.copy()
        from del_msh_numpy import Polyline
        self.vtx2framex_ini = Polyline.vtx2framex_from_vtx2xyz(vtx2xyz_ini)
        self.vtx2framex_def = self.vtx2framex_ini.copy()
        self.vtx2xyz_def = self.vtx2xyz_ini.copy()
        self.vtx2xyz_tmp = self.vtx2xyz_ini.copy()
        self.vtx2velo = numpy.zeros_like(self.vtx2xyz_ini)
        self.vtx2isfix = numpy.zeros(shape=(num_vtx,4), dtype=numpy.int32)
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
         from .del_fem_numpy import rod3_darboux_initialize_with_perturbation
         rod3_darboux_initialize_with_perturbation(
             self.vtx2xyz_def,
             self.vtx2framex_def,
             self.vtx2xyz_ini,
             self.vtx2framex_ini,
             self.vtx2isfix,
             pos_mag,
             framex_mag)

    def compute_rod_deformation_energy_grad_hessian(self, vtx2xyz_def, mdtt: float):
        self.w.fill(0.)
        self.dw.fill(0.)
        self.row2val.fill(0.)
        self.idx2val.fill(0.)
        from .del_fem_numpy import add_wdwddw_rod3_darboux
        add_wdwddw_rod3_darboux(
            self.vtx2xyz_ini,
            self.vtx2framex_ini,
            vtx2xyz_def,
            self.vtx2framex_def,
            mdtt,
            self.w,
            self.dw,
            self.row2idx,
            self.idx2col,
            self.row2val,
            self.idx2val)

    def apply_fix_bc(self):
        from .del_fem_numpy import block_sparse_apply_bc
        block_sparse_apply_bc(
            1.0,
            self.vtx2isfix,
            self.row2val,
            self.idx2val,
            self.row2idx,
            self.idx2col)
        from .del_fem_numpy import block_sparse_set_fixed_bc_to_rhs_vector
        block_sparse_set_fixed_bc_to_rhs_vector(
            self.vtx2isfix,
            self.dw)


    def update_solution_static(self, vtx2xyz_def):
        from .del_fem_numpy import conjugate_gradient
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
        from .del_fem_numpy import rod3_darboux_update_solution_hair
        rod3_darboux_update_solution_hair(
           vtx2xyz_def,
           self.vtx2framex_def,
           self.u_vec,
           self.vtx2isfix)


    def pull_vertex(self, vtx2xyz_def, i_vtx, goal_pos):
        stiff = 20.0
        self.row2val[i_vtx] += numpy.diag([stiff, stiff, stiff, 0.]).flatten()
        diff = self.vtx2xyz_def[i_vtx] - goal_pos
        self.dw[i_vtx] += numpy.append(diff, 0.) * stiff


    def solve_static(self, pull_vtx: Optional[Tuple[int, npt.NDArray[numpy.float32]]]):
        self.compute_rod_deformation_energy_grad_hessian(self.vtx2xyz_def, mdtt=0.0)
        if pull_vtx is not None:
            self.pull_vertex(self.vtx2xyz_def, *pull_vtx)
        self.apply_fix_bc()
        self.update_solution_static(self.vtx2xyz_def)


    def solve_dynamic(self, dt: float, pull_vtx: Optional[Tuple[int, npt.NDArray[numpy.float32]]]):
        num_vtx = self.vtx2xyz_ini.shape[0]
        for i_vtx in range(0,num_vtx):
            for i_dim in range(0, 3):
                if self.vtx2isfix[i_vtx, i_dim] != 0:
                    continue
                self.vtx2xyz_tmp[i_vtx,i_dim] = self.vtx2xyz_def[i_vtx,i_dim] + dt * self.vtx2velo[i_vtx, i_dim]

        from .del_fem_numpy import rod3_darboux_orthonormalize_framex_for_hair
        rod3_darboux_orthonormalize_framex_for_hair(self.vtx2framex_def, self.vtx2xyz_tmp)
        #
        self.compute_rod_deformation_energy_grad_hessian(self.vtx2xyz_tmp, mdtt=1.0/(dt*dt))
        if pull_vtx is not None:
            self.pull_vertex(self.vtx2xyz_tmp,*pull_vtx)
        self.apply_fix_bc()
        self.update_solution_static(self.vtx2xyz_tmp)
        for i_vtx in range(0,num_vtx):
            for i_dim in range(0, 3):
                if self.vtx2isfix[i_vtx, i_dim] != 0:
                    continue
                self.vtx2velo[i_vtx,i_dim] = (self.vtx2xyz_tmp[i_vtx, i_dim] - self.vtx2xyz_def[i_vtx, i_dim])/dt
                self.vtx2xyz_def[i_vtx, i_dim] = self.vtx2xyz_tmp[i_vtx, i_dim]
