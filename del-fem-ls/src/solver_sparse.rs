use crate::{sparse_ilu, sparse_square};
use num_traits::AsPrimitive;

/// solve linear system using the Confugate Gradient (CG) method
pub fn conjugate_gradient<T>(
    r_vec: &mut [T],
    u_vec: &mut Vec<T>,
    ap_vec: &mut Vec<T>,
    p_vec: &mut Vec<T>,
    conv_ratio_tol: T,
    max_iteration: usize,
    mat: &sparse_square::Matrix<T>,
) -> Vec<T>
where
    f32: AsPrimitive<T>,
    T: 'static
        + Copy
        + num_traits::Float
        + std::ops::MulAssign
        + std::ops::Mul
        + std::ops::AddAssign
        + std::cmp::PartialOrd
        + std::ops::Div<Output = T>
        + std::ops::Neg<Output = T>
        + std::fmt::Display
        + std::ops::Mul<Output = T>,
{
    use crate::slice::{add_scaled_vector, copy, dot, scale_and_add_vec, set_zero};
    {
        let n = r_vec.len();
        u_vec.resize(n, T::zero());
        ap_vec.resize(n, T::zero());
        p_vec.resize(n, T::zero());
    }
    let num_dim = r_vec.len() / mat.row2val.len();
    //
    let mut conv_hist = Vec::<T>::new();
    set_zero(u_vec);
    let mut sqnorm_res = dot(r_vec, r_vec);
    if sqnorm_res < 1.0e-20_f32.as_() {
        return conv_hist;
    }
    let inv_sqnorm_res_ini = T::one() / sqnorm_res;
    copy(p_vec, r_vec); // {p} = {r}  (set initial serch direction, copy value not reference)
    for _iitr in 0..max_iteration {
        if num_dim == 1 {
            // alpha = (r,r) / (p,Ap)
            sparse_square::mult_vec(ap_vec, T::zero(), T::one(), mat, p_vec); // {Ap_vec} = [mat]*{p_vec}
        } else {
            sparse_square::mult_mat(ap_vec, T::zero(), T::one(), mat, p_vec); // {Ap_vec} = [mat]*{p_vec}
        }
        let pap = dot(p_vec, ap_vec);
        assert!(pap >= T::zero());
        let alpha = sqnorm_res / pap;
        add_scaled_vector(u_vec, alpha, p_vec); // {u} = +alpha*{p} + {u} (update x)
        add_scaled_vector(r_vec, -alpha, ap_vec); // {r} = -alpha*{Ap} + {r}
        let sqnorm_res_new = dot(r_vec, r_vec);
        let conv_ratio = (sqnorm_res_new * inv_sqnorm_res_ini).sqrt();
        conv_hist.push(conv_ratio);
        if conv_ratio < conv_ratio_tol {
            return conv_hist;
        }
        {
            let beta = sqnorm_res_new / sqnorm_res; // beta = (r1,r1) / (r0,r0)
            sqnorm_res = sqnorm_res_new;
            scale_and_add_vec(p_vec, beta, r_vec); // {p} = {r} + beta*{p}
        }
    }
    conv_hist
}

/// solve a real-valued linear system using the conjugate gradient method with preconditioner
pub fn preconditioned_conjugate_gradient<T>(
    r_vec: &mut [T],
    x_vec: &mut Vec<T>,
    pr_vec: &mut Vec<T>,
    p_vec: &mut Vec<T>,
    conv_ratio_tol: T,
    max_nitr: usize,
    mat: &sparse_square::Matrix<T>,
    ilu: &sparse_ilu::Preconditioner<T>,
) -> Vec<T>
where
    T: 'static
        + Copy
        + std::ops::Mul
        + num_traits::Float
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::SubAssign,
    f32: AsPrimitive<T>,
{
    use crate::slice::{add_scaled_vector, copy, dot, scale_and_add_vec, set_zero};
    {
        let n = r_vec.len();
        x_vec.resize(n, T::zero());
        pr_vec.resize(n, T::zero());
        p_vec.resize(n, T::zero());
    }
    let num_dim = r_vec.len() / mat.num_blk;
    let mut conv_hist = Vec::<T>::new();

    set_zero(x_vec);

    let inv_sqnorm_res0 = {
        let sqnorm_res0 = dot(r_vec, r_vec); // DotX(r_vec, r_vec, N);
        conv_hist.push(sqnorm_res0.sqrt());
        if sqnorm_res0 < 1.0e-20_f32.as_() {
            return conv_hist;
        }
        T::one() / sqnorm_res0
    };

    // {Pr} = [P]{r}
    copy(pr_vec, r_vec); // std::vector<double> Pr_vec(r_vec, r_vec + N);

    if num_dim == 1 {
        sparse_ilu::solve_preconditioning_vec(pr_vec, ilu); // ilu.SolvePrecond(Pr_vec.data());
    } else {
        todo!();
    }

    // {p} = {Pr}
    copy(p_vec, pr_vec);

    // rPr = ({r},{Pr})
    let mut rpr = dot(r_vec, pr_vec); // DotX(r_vec, Pr_vec.data(), N);
    for _iitr in 0..max_nitr {
        // {Ap} = [A]{p}
        if num_dim == 1 {
            sparse_square::mult_vec(pr_vec, T::zero(), T::one(), mat, p_vec);
        } else {
            sparse_square::mult_mat(pr_vec, T::zero(), T::one(), mat, p_vec);
        }
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
            if num_dim == 1 {
                sparse_ilu::solve_preconditioning_vec(pr_vec, &ilu);
            } else {
                todo!();
            }
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
