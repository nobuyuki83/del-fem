import numpy
import numpy.typing as npt
from typing import Optional, Tuple
#
from del_fem_numpy.SparseSquare import SparseSquare


class Simulator:
    def __init__(self, vtx2xyz_ini):
        num_vtx = vtx2xyz_ini.shape[0]
        assert vtx2xyz_ini.shape == (num_vtx,3)
        from del_msh_numpy import Polyline
        self.vtx2framex_ini = Polyline.vtx2framex_from_vtx2xyz(vtx2xyz_ini)
        self.vtx2framex_def = self.vtx2framex_ini.copy()
        self.vtx2framex_tmp = self.vtx2framex_ini.copy()
        self.vtx2xyz_ini = vtx2xyz_ini.copy()
        self.vtx2xyz_def = self.vtx2xyz_ini.copy()
        self.vtx2xyz_tmp = self.vtx2xyz_ini.copy()
        self.vtx2velo = numpy.zeros(shape=(num_vtx,4), dtype=numpy.float32)
        self.vtx2isfix = numpy.zeros(shape=(num_vtx,4), dtype=numpy.int32)
        self.w = numpy.array(0., dtype=numpy.float32)
        self.dw = numpy.ndarray(shape=(num_vtx,4), dtype=numpy.float32)
        (row2idx, idx2col) = Polyline.vtx2vtx_rods(numpy.array([0, num_vtx], dtype=numpy.uint64))
        self.sparse = SparseSquare(row2idx, idx2col, 4)
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

    def compute_rod_deformation_energy_grad_hessian(self, vtx2xyz, vtx2framex, mdtt: float):
        self.w.fill(0.)
        self.dw.fill(0.)
        self.sparse.set_zero()
        from .del_fem_numpy import rod3_darboux_add_wdwddw
        rod3_darboux_add_wdwddw(
            self.vtx2xyz_ini,
            self.vtx2framex_ini,
            vtx2xyz,
            vtx2framex,
            mdtt,
            self.w,
            self.dw,
            self.sparse.row2idx,
            self.sparse.idx2col,
            self.sparse.row2val,
            self.sparse.idx2val)

    def compute_rod_total_energy(self, vtx2xyz, vtx2framex, mdtt) -> float:
        from .del_fem_numpy import rod3_darboux_add_w
        w_elastic = rod3_darboux_add_w(
            self.vtx2xyz_ini,
            self.vtx2framex_ini,
            vtx2xyz,
            vtx2framex)
        d = numpy.array([mdtt, mdtt, mdtt, 0.])
        w_kinetic = 0.5 * numpy.sum(d * numpy.sum(self.vtx2velo*self.vtx2velo, axis=0))
        return w_elastic + w_kinetic


    def apply_fix_bc(self):
        from .del_fem_numpy import block_sparse_apply_bc
        block_sparse_apply_bc(
            1.0,
            self.vtx2isfix,
            self.sparse.row2val,
            self.sparse.idx2val,
            self.sparse.row2idx,
            self.sparse.idx2col)
        from .del_fem_numpy import block_sparse_set_fixed_bc_to_rhs_vector
        block_sparse_set_fixed_bc_to_rhs_vector(
            self.vtx2isfix,
            self.dw)


    def update_solution_static(self, vtx2xyz, vtx2framex):
        from .del_fem_numpy import conjugate_gradient
        conv = conjugate_gradient(
            self.dw,
            self.u_vec,
            self.p_vec,
            self.ap_vec,
            self.sparse.row2idx,
            self.sparse.idx2col,
            self.sparse.idx2val,
            self.sparse.row2val)
        from .del_fem_numpy import rod3_darboux_update_solution_hair
        rod3_darboux_update_solution_hair(
           vtx2xyz,
           vtx2framex,
           self.u_vec,
           -1.,
           self.vtx2isfix)


    def pull_vertex(self, vtx2xyz, i_vtx, goal_pos, stiff_pull):
        self.sparse.row2val[i_vtx] += numpy.diag([stiff_pull, stiff_pull, stiff_pull, 0.]).flatten()
        diff1 = numpy.append(vtx2xyz[i_vtx] - goal_pos, 0.)
        self.dw[i_vtx] +=  diff1 * stiff_pull
        self.w += 0.5 * numpy.dot(diff1 * stiff_pull, diff1)


    def solve_static(self, pull_vtx: Optional[Tuple[int, npt.NDArray[numpy.float32]]]):
        stiff_pull = 20.0
        self.compute_rod_deformation_energy_grad_hessian(self.vtx2xyz_def, self.vtx2framex_def, mdtt=0.0)
        if pull_vtx is not None:
            self.pull_vertex(self.vtx2xyz_def, *pull_vtx, stiff_pull)
        self.apply_fix_bc()
        self.update_solution_static(self.vtx2xyz_def, self.vtx2framex_def)


    def solve_dynamic(self, dt: float, pull_vtx: Optional[Tuple[int, npt.NDArray[numpy.float32]]]):
        stiff_pull = 20.0
        mdtt = 1.0/(dt*dt)
        num_vtx = self.vtx2xyz_ini.shape[0]
        #
        self.vtx2xyz_tmp[:,:] = self.vtx2xyz_def
        self.vtx2framex_tmp[:,:] = self.vtx2framex_def
        from .del_fem_numpy import rod3_darboux_update_solution_hair
        rod3_darboux_update_solution_hair(
           self.vtx2xyz_tmp,
           self.vtx2framex_tmp,
           self.vtx2velo,
           dt,
           self.vtx2isfix)
        self.compute_rod_deformation_energy_grad_hessian(self.vtx2xyz_tmp, self.vtx2framex_tmp, mdtt=mdtt)
        if pull_vtx is not None:
            self.pull_vertex(self.vtx2xyz_tmp,*pull_vtx, stiff_pull)
        self.apply_fix_bc()
        self.update_solution_static(self.vtx2xyz_tmp, self.vtx2framex_tmp)
        #
        from .del_fem_numpy import rod3_darboux_compute_velocity
        rod3_darboux_compute_velocity(
            self.vtx2velo,
            dt,
            0.999,
            self.vtx2isfix,
            self.vtx2xyz_def,
            self.vtx2xyz_tmp,
            self.vtx2framex_def,
            self.vtx2framex_tmp)
        #
        w1 = self.compute_rod_total_energy(self.vtx2xyz_tmp, self.vtx2framex_tmp, mdtt=mdtt)
        if pull_vtx is not None:
            i_vtx, goal_pos = pull_vtx
            diff1 = self.vtx2xyz_tmp[i_vtx] - goal_pos
            w1 += 0.5 * stiff_pull * numpy.dot(diff1, diff1)
        # print(self.w, w1)
        if self.w * (1.1) < w1:
            print("instability", self.w, w1)
            self.vtx2velo[:,:] = 0.
            self.solve_static(pull_vtx)
        self.vtx2xyz_def[:,:] = self.vtx2xyz_tmp
        self.vtx2framex_def[:,:] = self.vtx2framex_tmp
