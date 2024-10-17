pub const ISTDIM2IJ: [[usize; 2]; 6] = [[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0]];

///!
/// compute energy density and its gradient & hessian w.r.t. right Cauchy-Green tensor given the displacement gradient tensor
/// * c1
/// * c2
/// * dudx - displacement gradient tensor
///
/// # Return
/// * density of elastic potential energy
/// * dWdC2 - gradient of energy density w.r.t. right Cauchy-Green tensor
/// * ddWddC2 - hessian of energy density w.r.t. right Cauchy-Green tensor
pub fn wr_dwrdc_ddwrddc_mooney_rivlin2_reduced<Real>(
    c1: Real,
    c2: Real,
    cauchy: &[[Real; 3]; 3],
) -> (Real, [Real; 6], [[Real; 6]; 6])
where
    Real: num_traits::Float + std::ops::AddAssign + std::ops::SubAssign + std::fmt::Debug,
{
    let zero = Real::zero();
    let one = Real::one();
    let two = one + one;
    let three = one + one + one;
    let half = one / two;
    let one3rd = one / three;
    let two3rd = two / three;

    // invariants of Cauchy-Green tensor
    let p1c = cauchy[0][0] + cauchy[1][1] + cauchy[2][2];
    let p2c =
        cauchy[0][0] * cauchy[1][1] + cauchy[0][0] * cauchy[2][2] + cauchy[1][1] * cauchy[2][2]
            - cauchy[0][1] * cauchy[1][0]
            - cauchy[0][2] * cauchy[2][0]
            - cauchy[1][2] * cauchy[2][1];
    let (p3c, c_inv) = del_geo_core::mat3_array_of_array::det_inv(&cauchy);
    let tmp1 = one / p3c.powf(one3rd);
    let tmp2 = one / p3c.powf(two3rd);
    let pi1c = p1c * tmp1; // 1st reduced invariant
    let pi2c = p2c * tmp2; // 2nd reduced invariant
    let wr = c1 * (pi1c - three) + c2 * (pi2c - three);
    let mut dwrdcv = [zero; 6];
    {
        // compute 2nd Piola-Kirchhoff tensor here
        let mut s = [[zero; 3]; 3]; // 2nd Piola-Kirchhoff tensor
        for (idim, jdim) in itertools::iproduct!(0..3, 0..3) {
            s[idim][jdim] = -c2 * tmp2 * cauchy[idim][jdim]
                - one3rd * (c1 * pi1c + c2 * two * pi2c) * c_inv[idim][jdim];
        }
        {
            let dtmp1 = c1 * tmp1 + c2 * tmp2 * p1c;
            s[0][0] += dtmp1;
            s[1][1] += dtmp1;
            s[2][2] += dtmp1;
        }
        {
            // 2nd piola-kirchhoff tensor is symmetric. Here extracting 6 independent elements.
            dwrdcv[0] = s[ISTDIM2IJ[0][0]][ISTDIM2IJ[0][1]];
            dwrdcv[1] = s[ISTDIM2IJ[1][0]][ISTDIM2IJ[1][1]];
            dwrdcv[2] = s[ISTDIM2IJ[2][0]][ISTDIM2IJ[2][1]];
            dwrdcv[3] = s[ISTDIM2IJ[3][0]][ISTDIM2IJ[3][1]];
            dwrdcv[4] = s[ISTDIM2IJ[4][0]][ISTDIM2IJ[4][1]];
            dwrdcv[5] = s[ISTDIM2IJ[5][0]][ISTDIM2IJ[5][1]];
        }
    }

    // computing constituive tensor from here
    let mut ddwrddc = [[[[zero; 3]; 3]; 3]; 3];
    for (idim, jdim, kdim, ldim) in itertools::iproduct!(0..3, 0..3, 0..3, 0..3) {
        let mut tmp = zero;
        tmp += c1 * tmp1 / three
            * (c_inv[idim][jdim] * c_inv[kdim][ldim] * p1c / three
                + c_inv[idim][kdim] * c_inv[ldim][jdim] * p1c * half
                + c_inv[idim][ldim] * c_inv[kdim][jdim] * p1c * half);
        tmp += c2
            * tmp2
            * two3rd
            * (c_inv[idim][jdim] * c_inv[kdim][ldim] * p2c * two3rd
                + c_inv[idim][jdim] * cauchy[kdim][ldim]
                + cauchy[idim][jdim] * c_inv[kdim][ldim]
                + c_inv[idim][kdim] * c_inv[jdim][ldim] * p2c * half
                + c_inv[idim][ldim] * c_inv[jdim][kdim] * p2c * half);
        ddwrddc[idim][jdim][kdim][ldim] += tmp;
    }

    for (idim, jdim) in itertools::iproduct!(0..3, 0..3) {
        let dtmp1 = c1 * tmp1 / three * c_inv[idim][jdim];
        for kdim in 0..3 {
            ddwrddc[idim][jdim][kdim][kdim] -= dtmp1;
            ddwrddc[kdim][kdim][idim][jdim] -= dtmp1;
        }
        let dtmp2 = c2 * tmp2 * p1c * two3rd * c_inv[idim][jdim];
        for kdim in 0..3 {
            ddwrddc[idim][jdim][kdim][kdim] -= dtmp2;
            ddwrddc[kdim][kdim][idim][jdim] -= dtmp2;
        }
        ddwrddc[idim][idim][jdim][jdim] += c2 * tmp2;
        ddwrddc[idim][jdim][jdim][idim] -= half * c2 * tmp2;
        ddwrddc[idim][jdim][idim][jdim] -= half * c2 * tmp2;
    }

    let mut ddwrddcv = [[zero; 6]; 6];
    {
        // Extracting independent components in the constitutive tensor
        for (istdim, jstdim) in itertools::iproduct!(0..6, 0..6) {
            let idim = ISTDIM2IJ[istdim][0];
            let jdim = ISTDIM2IJ[istdim][1];
            let kdim = ISTDIM2IJ[jstdim][0];
            let ldim = ISTDIM2IJ[jstdim][1];
            ddwrddcv[istdim][jstdim] = ddwrddc[idim][jdim][kdim][ldim];
        }
    }
    return (wr, dwrdcv, ddwrddcv);
}

#[test]
pub fn test_hoge() {
    let a1 = 0.8f64;
    let a2 = 0.3f64;
    let cv0: [f64; 6] = [1., 0.9, 1.1, -0.1, -0.2, -0.3];
    let c0 = crate::dudx::tensor3_from_symmetric_vector_param(&cv0);
    let (w0, dw0, ddw0) = wr_dwrdc_ddwrddc_mooney_rivlin2_reduced(a1, a2, &c0);
    let eps = 1.0e-6;
    for i_dim in 0..6 {
        let mut c1 = c0;
        c1[ISTDIM2IJ[i_dim][0]][ISTDIM2IJ[i_dim][1]] += eps;
        let (w1, _dw1, _ddw) = wr_dwrdc_ddwrddc_mooney_rivlin2_reduced(a1, a2, &c1);
        {
            let v_num = (w1 - w0) / eps;
            let v_ana = dw0[i_dim];
            assert!((v_num - v_ana).abs() < 1.0e-6);
        }
    }
    // check symmetrized derivative
    for i_dim in 0..6 {
        let mut c1 = c0;
        c1[ISTDIM2IJ[i_dim][0]][ISTDIM2IJ[i_dim][1]] += eps * 0.5;
        c1[ISTDIM2IJ[i_dim][1]][ISTDIM2IJ[i_dim][0]] += eps * 0.5;
        let (_w1, dw1, _ddw) = wr_dwrdc_ddwrddc_mooney_rivlin2_reduced(a1, a2, &c1);
        for j_dim in 0..6 {
            let v_num = (dw1[j_dim] - dw0[j_dim]) / eps;
            let v_ana = ddw0[i_dim][j_dim];
            assert!((v_num - v_ana).abs() < 5.0e-6);
        }
    }
}
