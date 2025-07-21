//! sparse matrix class and functions

/// sparse matrix class
/// Compressed Row Storage (CRS) data structure
/// * `num_blk` - number of row and col blocks
pub struct Matrix<MAT> {
    pub num_blk: usize,
    pub row2idx: Vec<usize>,
    pub idx2col: Vec<usize>,
    pub idx2val: Vec<MAT>,
    pub row2val: Vec<MAT>,
}

/*
impl<MAT> Matrix<MAT>
where
    MAT: 'static
        + num_traits::Zero
        + std::default::Default
        + std::ops::AddAssign // merge
        + Copy
{
    pub fn new() -> Self {
        Matrix {
            num_blk: 0,
            row2idx: vec![0],
            idx2col: Vec::<usize>::new(),
            idx2val: Vec::<MAT>::new(),
            row2val: Vec::<MAT>::new(),
        }
    }

    pub fn clone(&self) -> Self {
        Matrix {
            num_blk: self.num_blk,
            row2idx: self.row2idx.clone(),
            idx2col: self.idx2col.clone(),
            idx2val: self.idx2val.clone(),
            row2val: self.row2val.clone(),
        }
    }

    /// set non-zero pattern
    pub fn symbolic_initialization(&mut self, row2idx: &[usize], idx2col: &[usize]) {
        self.num_blk = row2idx.len() - 1;
        self.row2idx = row2idx.to_vec();
        self.idx2col = idx2col.to_vec();
        let num_idx = self.row2idx[self.num_blk];
        assert_eq!(num_idx, idx2col.len());
        self.idx2val.resize_with(num_idx, Default::default);
        self.row2val.resize_with(self.num_blk, Default::default);
    }

    /// set zero to all the values
    pub fn set_zero(&mut self) {
        assert_eq!(self.idx2val.len(), self.idx2col.len());
        for m in self.row2val.iter_mut() {
            m.set_zero()
        }
        for m in self.idx2val.iter_mut() {
            m.set_zero()
        }
    }

    /// merge element-wise matrix to sparse matrix
    pub fn merge(
        &mut self,
        node2row: &[usize],
        node2col: &[usize],
        emat: &[MAT],
        merge_buffer: &mut Vec<usize>,
    ) {
        assert_eq!(emat.len(), node2row.len() * node2col.len());
        merge_buffer.resize(self.num_blk, usize::MAX);
        let col2idx = merge_buffer;
        for inode in 0..node2row.len() {
            let i_row = node2row[inode];
            assert!(i_row < self.num_blk);
            for ij_idx in self.row2idx[i_row]..self.row2idx[i_row + 1] {
                assert!(ij_idx < self.idx2col.len());
                let j_col = self.idx2col[ij_idx];
                col2idx[j_col] = ij_idx;
            }
            for jnode in 0..node2col.len() {
                let j_col = node2col[jnode];
                assert!(j_col < self.num_blk);
                if i_row == j_col {
                    // Marge Diagonal
                    self.row2val[i_row] += emat[inode * node2col.len() + jnode];
                } else {
                    // Marge Non-Diagonal
                    assert!(col2idx[j_col] < self.idx2col.len());
                    let ij_idx = col2idx[j_col];
                    assert_eq!(self.idx2col[ij_idx], j_col);
                    self.idx2val[ij_idx] += emat[inode * node2col.len() + jnode];
                }
            }
            for ij_idx in self.row2idx[i_row]..self.row2idx[i_row + 1] {
                assert!(ij_idx < self.idx2col.len());
                let j_col = self.idx2col[ij_idx];
                col2idx[j_col] = usize::MAX;
            }
        }
    }
}

/// generalized matrix-vector multiplication
/// where matrix is sparse (not block) matrix
/// `{y_vec} <- \alpha * [a_mat] * {x_vec} + \beta * {y_vec}`
pub fn mult_vec<T>(y_vec: &mut Vec<T>, beta: T, alpha: T, a_mat: &Matrix<T>, x_vec: &Vec<T>)
where
    T: std::ops::MulAssign // *=
        + std::ops::Mul<Output = T> // *
        + std::ops::AddAssign // +=
        + 'static
        + Copy, // =
    f32: num_traits::AsPrimitive<T>,
{
    assert_eq!(y_vec.len(), a_mat.num_blk);
    for m in y_vec.iter_mut() {
        *m *= beta;
    }
    for iblk in 0..a_mat.num_blk {
        for icrs in a_mat.row2idx[iblk]..a_mat.row2idx[iblk + 1] {
            assert!(icrs < a_mat.idx2col.len());
            let jblk0 = a_mat.idx2col[icrs];
            assert!(jblk0 < a_mat.num_blk);
            y_vec[iblk] += alpha * a_mat.idx2val[icrs] * x_vec[jblk0];
        }
        y_vec[iblk] += alpha * a_mat.row2val[iblk] * x_vec[iblk];
    }
}

pub fn mult_mat<T>(y_mat: &mut [T], beta: T, alpha: T, a_mat: &Matrix<T>, x_mat: &[T])
where
    T: std::ops::MulAssign // *=
        + std::ops::Mul<Output = T> // *
        + std::ops::AddAssign // +=
        + 'static
        + Copy, // =
    f32: num_traits::AsPrimitive<T>,
{
    assert_eq!(y_mat.len(), x_mat.len());
    assert!(y_mat.len() > 0);
    let num_row = a_mat.row2idx.len() - 1;
    let num_dim = y_mat.len() / num_row;
    assert_eq!(num_dim * num_row, y_mat.len());
    assert_eq!(y_mat.len(), num_dim * num_row);
    for val_y in y_mat.iter_mut() {
        *val_y *= beta;
    }
    for i_row in 0..num_row {
        for idx in a_mat.row2idx[i_row]..a_mat.row2idx[i_row + 1] {
            let j_col = a_mat.idx2col[idx];
            for y in 0..num_dim {
                y_mat[i_row * num_dim + y] +=
                    alpha * a_mat.idx2val[idx] * x_mat[j_col * num_dim + y];
            }
        }
        for y in 0..num_dim {
            y_mat[i_row * num_dim + y] += alpha * a_mat.row2val[i_row] * x_mat[i_row * num_dim + y];
        }
    }
}

#[test]
fn test_scalar() {
    let mut sparse = crate::sparse_square::Matrix::<f32>::new();
    let colind = vec![0, 2, 5, 8, 10];
    let rowptr = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    sparse.symbolic_initialization(&colind, &rowptr);
    sparse.set_zero();
    {
        let emat = [1., 0., 0., 1.];
        let mut tmp_buffer = Vec::<usize>::new();
        sparse.merge(&[0, 1], &[0, 1], &emat, &mut tmp_buffer);
    }
    let nblk = colind.len() - 1;
    let mut rhs = Vec::<f32>::new();
    rhs.resize(nblk, Default::default());
    let mut lhs = Vec::<f32>::new();
    lhs.resize(nblk, Default::default());
    mult_vec(&mut lhs, 1.0, 1.0, &sparse, &rhs);
}
 */

impl<T, const NN: usize> Matrix<[T; NN]>
where
    T: num_traits::Float,
{
    fn set_fixed_bc_dia<const N: usize>(&mut self, val_dia: T, bc_flag: &[[i32; N]]) {
        let num_blk = bc_flag.len();
        assert_eq!(bc_flag.len(), self.row2val.len());
        for i_blk in 0..num_blk {
            // set diagonal
            for i_dim in 0..N {
                if bc_flag[i_blk][i_dim] == 0 {
                    continue;
                };
                for j_dim in 0..N {
                    self.row2val[i_blk][i_dim + N * j_dim] = T::zero();
                    self.row2val[i_blk][j_dim + N * i_dim] = T::zero();
                }
                self.row2val[i_blk][i_dim + N * i_dim] = val_dia;
            }
        }
    }

    fn set_fixed_bc_row<const N: usize>(&mut self, bc_flag: &[[i32; N]]) {
        assert_eq!(bc_flag.len(), self.num_blk);
        for i_blk in 0..self.num_blk {
            // set row
            for idx in self.row2idx[i_blk]..self.row2idx[i_blk + 1] {
                for i_dim in 0..N {
                    if bc_flag[i_blk][i_dim] == 0 {
                        continue;
                    };
                    for j_dim in 0..N {
                        self.idx2val[idx][i_dim + N * j_dim] = T::zero();
                    }
                }
            }
        }
    }

    fn set_fixed_bc_col<const N: usize>(&mut self, bc_flag: &[[i32; N]]) {
        for idx in 0..self.idx2col.len() {
            let j_blk1 = self.idx2col[idx];
            for j_dim in 0..N {
                if bc_flag[j_blk1][j_dim] == 0 {
                    continue;
                };
                for i_dim in 0..N {
                    self.idx2val[idx][i_dim + N * j_dim] = T::zero();
                }
            }
        }
    }

    pub fn set_fixed_dof<const N: usize>(&mut self, val_dia: T, bc_flag: &[[i32; N]]) {
        self.set_fixed_bc_dia::<N>(val_dia, bc_flag);
        self.set_fixed_bc_col::<N>(bc_flag);
        self.set_fixed_bc_row::<N>(bc_flag);
    }

    /// generalized matrix-vector multiplication
    /// where matrix is sparse (not block) matrix
    /// `{y_vec} <- \alpha * [a_mat] * {x_vec} + \beta * {y_vec}`
    pub fn mult_vec<const N: usize>(
        &self,
        y_vec: &mut [[T; N]],
        beta: T,
        alpha: T,
        x_vec: &[[T; N]],
    ) where
        T: num_traits::Float,
    {
        use del_geo_core::matn_col_major;
        use del_geo_core::vecn::VecN;
        assert_eq!(y_vec.len(), self.num_blk);
        for m in y_vec.iter_mut() {
            del_geo_core::vecn::scale_in_place(m, beta);
        }
        for i_blk in 0..self.num_blk {
            for idx in self.row2idx[i_blk]..self.row2idx[i_blk + 1] {
                assert!(idx < self.idx2col.len());
                let j_blk = self.idx2col[idx];
                assert!(j_blk < self.num_blk);
                let a = matn_col_major::mult_vec(&self.idx2val[idx], &x_vec[j_blk]).scale(alpha);
                del_geo_core::vecn::add_in_place(&mut y_vec[i_blk], &a);
            }
            {
                let a = matn_col_major::mult_vec(&self.row2val[i_blk], &x_vec[i_blk]).scale(alpha);
                del_geo_core::vecn::add_in_place(&mut y_vec[i_blk], &a);
            }
        }
    }

    /// set zero to all the values
    pub fn set_zero(&mut self) {
        assert_eq!(self.idx2val.len(), self.idx2col.len());
        for m in self.row2val.iter_mut() {
            m.iter_mut().for_each(|v| *v = T::zero());
        }
        for m in self.idx2val.iter_mut() {
            m.iter_mut().for_each(|v| *v = T::zero());
        }
    }

    pub fn merge_for_array_blk<const NNODE: usize>(
        &mut self,
        emat: &[[[T; NN]; NNODE]; NNODE],
        node2vtx: &[usize; NNODE],
        col2idx: &mut Vec<usize>,
    ) {
        col2idx.resize(self.num_blk, usize::MAX);
        for i_node in 0..NNODE {
            let i_vtx = node2vtx[i_node];
            for idx in self.row2idx[i_vtx]..self.row2idx[i_vtx + 1] {
                let j_vtx = self.idx2col[idx];
                col2idx[j_vtx] = idx;
            }
            for j_node in 0..NNODE {
                if i_node == j_node {
                    del_geo_core::matn_col_major::add_in_place(
                        &mut self.row2val[i_vtx],
                        &emat[i_node][j_node],
                    );
                } else {
                    let j_vtx = node2vtx[j_node];
                    let idx0 = col2idx[j_vtx];
                    assert_ne!(idx0, usize::MAX);
                    del_geo_core::matn_col_major::add_in_place(
                        &mut self.idx2val[idx0],
                        &emat[i_node][j_node],
                    );
                }
            }
            for idx in self.row2idx[i_vtx]..self.row2idx[i_vtx + 1] {
                let j_vtx = self.idx2col[idx];
                col2idx[j_vtx] = usize::MAX;
            }
        }
    }
}

/// solve linear system using the Conjugate Gradient (CG) method
pub fn conjugate_gradient0<T>(
    r_vec: &mut [[T; 2]],
    u_vec: &mut Vec<[T; 2]>,
    ap_vec: &mut Vec<[T; 2]>,
    p_vec: &mut Vec<[T; 2]>,
    conv_ratio_tol: T,
    max_iteration: usize,
    mat: &Matrix<[T; 4]>,
) -> Vec<T>
where
    f32: num_traits::AsPrimitive<T>,
    T: 'static + Copy + num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
    {
        let n = r_vec.len();
        u_vec.resize(n, [T::zero(); 2]);
        ap_vec.resize(n, [T::zero(); 2]);
        p_vec.resize(n, [T::zero(); 2]);
    }
    let _num_dim = r_vec.len() / mat.row2val.len();
    //
    let mut conv_hist = Vec::<T>::new();
    crate::slice_of_array::set_zero(u_vec);
    let mut sqnorm_res = crate::slice_of_array::dot(r_vec, r_vec);
    if sqnorm_res < T::epsilon() {
        return conv_hist;
    }
    let inv_sqnorm_res_ini = T::one() / sqnorm_res;
    crate::slice_of_array::copy(p_vec, r_vec); // {p} = {r}  (set initial serch direction, copy value not reference)
    for _iitr in 0..max_iteration {
        // alpha = (r,r) / (p,Ap)
        mat.mult_vec::<2>(ap_vec, T::zero(), T::one(), p_vec); // {Ap_vec} = [mat]*{p_vec}
        let pap = crate::slice_of_array::dot(p_vec, ap_vec);
        assert!(pap >= T::zero(), "{pap}");
        let alpha = sqnorm_res / pap;
        crate::slice_of_array::add_scaled_vector(u_vec, alpha, p_vec); // {u} = +alpha*{p} + {u} (update x)
        crate::slice_of_array::add_scaled_vector(r_vec, -alpha, ap_vec); // {r} = -alpha*{Ap} + {r}
        let sqnorm_res_new = crate::slice_of_array::dot(r_vec, r_vec);
        let conv_ratio = (sqnorm_res_new * inv_sqnorm_res_ini).sqrt();
        conv_hist.push(conv_ratio);
        if conv_ratio < conv_ratio_tol {
            return conv_hist;
        }
        {
            let beta = sqnorm_res_new / sqnorm_res; // beta = (r1,r1) / (r0,r0)
            sqnorm_res = sqnorm_res_new;
            crate::slice_of_array::scale_and_add_vec(p_vec, beta, r_vec); // {p} = {r} + beta*{p}
        }
    }
    conv_hist
}

/// solve a real-valued linear system using the conjugate gradient method with preconditioner
pub fn preconditioned_conjugate_gradient<T, const N: usize, const NN: usize>(
    r_vec: &mut [[T; 2]],
    x_vec: &mut Vec<[T; 2]>,
    pr_vec: &mut Vec<[T; 2]>,
    p_vec: &mut Vec<[T; 2]>,
    conv_ratio_tol: T,
    max_nitr: usize,
    mat: &crate::sparse_square::Matrix<[T; 4]>,
    ilu: &crate::sparse_ilu::Preconditioner<[T; 4]>,
) -> Vec<T>
where
    T: num_traits::Float + std::fmt::Debug,
{
    use crate::slice_of_array::{add_scaled_vector, copy, dot, scale_and_add_vec, set_zero};
    {
        let n = r_vec.len();
        x_vec.resize(n, [T::zero(); 2]);
        pr_vec.resize(n, [T::zero(); 2]);
        p_vec.resize(n, [T::zero(); 2]);
    }
    assert_eq!(r_vec.len(), mat.num_blk);
    let mut conv_hist = Vec::<T>::new();

    set_zero(x_vec);

    let inv_sqnorm_res0 = {
        let sqnorm_res0 = dot(r_vec, r_vec); // DotX(r_vec, r_vec, N);
        conv_hist.push(sqnorm_res0.sqrt());
        if sqnorm_res0 < T::epsilon() {
            return conv_hist;
        }
        T::one() / sqnorm_res0
    };

    // {Pr} = [P]{r}
    copy(pr_vec, r_vec); // std::vector<double> Pr_vec(r_vec, r_vec + N);

    crate::sparse_ilu::solve_preconditioning_vec(pr_vec, ilu); // ilu.SolvePrecond(Pr_vec.data());

    // {p} = {Pr}
    copy(p_vec, pr_vec);

    // rPr = ({r},{Pr})
    let mut rpr = dot(r_vec, pr_vec); // DotX(r_vec, Pr_vec.data(), N);
    for _iitr in 0..max_nitr {
        // {Ap} = [A]{p}
        mat.mult_vec(pr_vec, T::zero(), T::one(), p_vec);
        {
            // alpha = ({r},{Pr})/({p},{Ap})
            let pap = dot(p_vec, pr_vec);
            let alpha = rpr / pap;
            add_scaled_vector(r_vec, -alpha, pr_vec); // {r} = -alpha*{Ap} + {r}
            add_scaled_vector(x_vec, alpha, p_vec); // {x} = +alpha*{p} + {x}
        }
        {
            // Converge Judgement
            let sqnorm_res = dot(r_vec, r_vec); // DotX(r_vec, r_vec, N);
            conv_hist.push(sqnorm_res.sqrt());
            let conv_ratio = (sqnorm_res * inv_sqnorm_res0).sqrt();
            if conv_ratio < conv_ratio_tol {
                return conv_hist;
            }
        }
        {
            // calc beta
            copy(pr_vec, r_vec);
            // {Pr} = [P]{r}
            crate::sparse_ilu::solve_preconditioning_vec(pr_vec, ilu);
            // rPr1 = ({r},{Pr})
            let rpr1 = dot(r_vec, pr_vec);
            // beta = rPr1/rPr
            let beta = rpr1 / rpr;
            rpr = rpr1;
            // {p} = {Pr} + beta*{p}
            scale_and_add_vec(p_vec, beta, pr_vec);
        }
    }
    {
        // Converge Judgement
        let sq_norm_res = dot(r_vec, r_vec); // DotX(r_vec, r_vec, N);
        conv_hist.push(sq_norm_res.sqrt());
    }
    conv_hist
}
