//! Linear system solver class

use num_traits::AsPrimitive;

/// class of linear system solver
/// * `sparse` - sparse square coefficient matrix
/// * `r_vec` - residual vector (i.e., rhs vector)
pub struct Solver<T> {
    pub sparse: crate::sparse_square::Matrix<T>,
    pub ilu: crate::sparse_ilu::Preconditioner<T>,
    pub merge_buffer: Vec<usize>,
    pub r_vec: Vec<T>,
    pub u_vec: Vec<T>,
    pub conv: Vec<T>,
    pub conv_ratio_tol: T,
    pub max_num_iteration: usize,
    pub ap_vec: Vec<T>,
    pub p_vec: Vec<T>,
}

impl<T> Solver<T>
where
    T: 'static
        + Copy
        + num_traits::Float
        + std::default::Default
        + std::ops::AddAssign
        + std::fmt::Display
        + std::ops::MulAssign
        + std::ops::SubAssign,
    f32: num_traits::AsPrimitive<T>,
{
    pub fn new() -> Self {
        Solver {
            sparse: crate::sparse_square::Matrix::<T>::new(),
            ilu: crate::sparse_ilu::Preconditioner::new(),
            merge_buffer: Vec::<usize>::new(),
            ap_vec: Vec::<T>::new(),
            p_vec: Vec::<T>::new(),
            r_vec: Vec::<T>::new(),
            u_vec: Vec::<T>::new(),
            conv: Vec::<T>::new(),
            conv_ratio_tol: 1.0e-5_f32.as_(),
            max_num_iteration: 100,
        }
    }

    pub fn initialize(&mut self, colind: &Vec<usize>, rowptr: &Vec<usize>) {
        self.sparse.symbolic_initialization(&colind, &rowptr);
        //self.ilu.initialize(&self.sparse);
        let nblk = colind.len() - 1;
        self.r_vec.resize(nblk, T::zero());
        self.ilu.initialize_ilu0(&self.sparse);
        //self.ilu.Initialize_ILUk(&self.sparse, 0);
        //self.ilu.initialize_iluk(&self.sparse, 10000);
        //self.ilu.initialize_full(self.sparse.num_blk);
    }

    /// set zero to the matrix
    pub fn begin_merge(&mut self) {
        self.sparse.set_zero();
        let nblk = self.sparse.num_blk;
        self.r_vec.resize(nblk, T::zero());
        self.r_vec.iter_mut().for_each(|v| v.set_zero());
    }

    pub fn end_merge(&mut self) {
        // apply boundary condition here
        crate::sparse_ilu::copy_value(&mut self.ilu, &self.sparse);
        crate::sparse_ilu::decompose(&mut self.ilu);
    }

    pub fn solve_cg(&mut self) {
        self.conv = crate::solver_sparse::conjugate_gradient(
            &mut self.r_vec,
            &mut self.u_vec,
            &mut self.ap_vec,
            &mut self.p_vec,
            self.conv_ratio_tol,
            self.max_num_iteration,
            &self.sparse,
        );
    }

    pub fn solve_pcg(&mut self) {
        self.conv = crate::solver_sparse::preconditioned_conjugate_gradient(
            &mut self.r_vec,
            &mut self.u_vec,
            &mut self.ap_vec,
            &mut self.p_vec,
            self.conv_ratio_tol,
            self.max_num_iteration,
            &self.sparse,
            &self.ilu,
        );
    }

    pub fn clone(&self) -> Self {
        Solver {
            sparse: self.sparse.clone(),
            ilu: self.ilu.clone(),
            merge_buffer: self.merge_buffer.clone(),
            ap_vec: self.ap_vec.clone(),
            p_vec: self.p_vec.clone(),
            r_vec: self.r_vec.clone(),
            u_vec: self.u_vec.clone(),
            conv: self.conv.clone(),
            conv_ratio_tol: self.conv_ratio_tol,
            max_num_iteration: 100,
        }
    }
}
