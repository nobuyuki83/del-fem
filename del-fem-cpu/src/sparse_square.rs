//! sparse matrix class and functions

pub fn set_fixed_bc<T, const N: usize, const NN: usize>(
    val_dia: T,
    bc_flag: &[[i32; N]],
    row2val: &mut [[T; NN]],
    idx2val: &mut [[T; NN]],
    row2idx: &[usize],
    idx2col: &[usize],
) where
    T: num_traits::Float,
{
    let num_blk = bc_flag.len();
    assert_eq!(bc_flag.len(), row2val.len());
    for i_blk in 0..num_blk {
        // set diagonal
        for i_dim in 0..N {
            if bc_flag[i_blk][i_dim] == 0 {
                continue;
            };
            for j_dim in 0..N {
                row2val[i_blk][i_dim + N * j_dim] = T::zero();
                row2val[i_blk][j_dim + N * i_dim] = T::zero();
            }
            row2val[i_blk][i_dim + N * i_dim] = val_dia;
        }
    }
    //
    assert_eq!(bc_flag.len(), num_blk);
    for i_blk in 0..num_blk {
        // set row
        for idx in row2idx[i_blk]..row2idx[i_blk + 1] {
            for i_dim in 0..N {
                if bc_flag[i_blk][i_dim] == 0 {
                    continue;
                };
                for j_dim in 0..N {
                    idx2val[idx][i_dim + N * j_dim] = T::zero();
                }
            }
        }
    }
    //
    for idx in 0..idx2col.len() {
        let j_blk1 = idx2col[idx];
        for j_dim in 0..N {
            if bc_flag[j_blk1][j_dim] == 0 {
                continue;
            };
            for i_dim in 0..N {
                idx2val[idx][i_dim + N * j_dim] = T::zero();
            }
        }
    }
}

pub fn set_fix_dof_to_rhs_vector<T, const N: usize>(blk2rhs: &mut [[T; N]], blk2isfix: &[[i32; N]])
where
    T: num_traits::Float,
{
    let num_vtx = blk2rhs.len();
    for i_vtx in 0..num_vtx {
        for i_dof in 0..N {
            if blk2isfix[i_vtx][i_dof] == 0 {
                continue;
            }
            blk2rhs[i_vtx][i_dof] = T::zero();
        }
    }
}

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

impl<T, const NN: usize> Matrix<[T; NN]>
where
    T: num_traits::Float,
{
    pub fn new() -> Self {
        Matrix {
            num_blk: 0,
            row2idx: vec![0],
            idx2col: Vec::<usize>::new(),
            idx2val: Vec::<[T; NN]>::new(),
            row2val: Vec::<[T; NN]>::new(),
        }
    }

    pub fn as_ref_mut(&mut self) -> MatrixRefMut<T, NN> {
        MatrixRefMut {
            num_blk: self.num_blk,
            row2idx: &self.row2idx,
            idx2col: &self.idx2col,
            idx2val: &mut self.idx2val,
            row2val: &mut self.row2val,
        }
    }

    pub fn as_ref(&self) -> MatrixRef<T, NN> {
        MatrixRef {
            num_blk: self.num_blk,
            row2idx: &self.row2idx,
            idx2col: &self.idx2col,
            idx2val: &self.idx2val,
            row2val: &self.row2val,
        }
    }

    pub fn from_vtx2vtx(vtx2idx: &[usize], idx2vtx: &[usize]) -> Self {
        let num_blk = vtx2idx.len() - 1;
        let num_idx = vtx2idx[num_blk];
        Self {
            num_blk,
            row2idx: vtx2idx.to_vec(),
            idx2col: idx2vtx.to_vec(),
            idx2val: vec![[T::zero(); NN]; num_idx],
            row2val: vec![[T::zero(); NN]; num_blk],
        }
    }

    pub fn set_fixed_dof<const N: usize>(&mut self, val_dia: T, blk2isfix: &[[i32; N]]) {
        set_fixed_bc(
            val_dia,
            blk2isfix,
            &mut self.row2val,
            &mut self.idx2val,
            &self.row2idx,
            &self.idx2col,
        );
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
        self.row2val.fill([T::zero(); NN]);
        self.idx2val.fill([T::zero(); NN]);
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
pub fn conjugate_gradient<T, const N: usize, const NN: usize>(
    r_vec: &mut [[T; N]],
    u_vec: &mut [[T; N]],
    ap_vec: &mut [[T; N]],
    p_vec: &mut [[T; N]],
    conv_ratio_tol: T,
    max_iteration: usize,
    mat: MatrixRef<T, NN>,
) -> Vec<T>
where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
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
        mat.mult_vec::<N>(ap_vec, T::zero(), T::one(), p_vec); // {Ap_vec} = [mat]*{p_vec}
        let pap = crate::slice_of_array::dot(p_vec, ap_vec);
        // assert!(pap >= T::zero(), "{pap}");
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

pub struct MatrixRefMut<'a, T, const NN: usize> {
    pub num_blk: usize,
    pub row2idx: &'a [usize],
    pub idx2col: &'a [usize],
    pub idx2val: &'a mut [[T; NN]],
    pub row2val: &'a mut [[T; NN]],
}

impl<'a, T, const NN: usize> MatrixRefMut<'a, T, NN>
where
    T: num_traits::Float,
{
    pub fn set_zero(&mut self) {
        assert_eq!(self.idx2val.len(), self.idx2col.len());
        self.row2val.fill([T::zero(); NN]);
        self.idx2val.fill([T::zero(); NN]);
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

pub struct MatrixRef<'a, T, const NN: usize> {
    pub num_blk: usize,
    pub row2idx: &'a [usize],
    pub idx2col: &'a [usize],
    pub idx2val: &'a [[T; NN]],
    pub row2val: &'a [[T; NN]],
}

impl<'a, T, const NN: usize> MatrixRef<'a, T, NN>
where
    T: num_traits::Float,
{
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
}
