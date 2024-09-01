pub fn def_grad_tensor<const NDIM: usize, Real>(dudx: &[[Real; NDIM]; NDIM]) -> [[Real; NDIM]; NDIM]
where
    Real: num_traits::Float + std::ops::AddAssign,
{
    let one = Real::one();
    let mut z = [[Real::zero(); NDIM]; NDIM];
    for idim in 0..NDIM {
        for jdim in 0..NDIM {
            z[idim][jdim] = dudx[idim][jdim];
        }
        z[idim][idim] += one;
    }
    z
}

/// C=F^TF = (Z + I)^T(Z+I) = Z^TZ + Z + Z^T + I
pub fn right_cauchy_green_tensor<const NDIM: usize, Real>(
    dudx: &[[Real; NDIM]; NDIM],
) -> [[Real; NDIM]; NDIM]
where
    Real: num_traits::Float + std::ops::AddAssign,
{
    let mut c = [[Real::zero(); NDIM]; NDIM];
    for idim in 0..NDIM {
        for jdim in 0..NDIM {
            c[idim][jdim] = dudx[idim][jdim] + dudx[jdim][idim];
            for kdim in 0..NDIM {
                c[idim][jdim] += dudx[kdim][idim] * dudx[kdim][jdim];
            }
        }
        c[idim][idim] += Real::one();
    }
    c
}

const ISTDIM2IJ: [[usize; 2]; 6] = [[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0]];

pub fn wr_dwrdc_ddwrddc_energy_density_sqr_compression<Real>(
    c: &[[Real; 3]; 3],
) -> (Real, [Real; 6], [[Real; 6]; 6])
where
    Real: num_traits::Float + std::ops::AddAssign,
{
    let zero = Real::zero();
    let one = Real::one();
    let two = one + one;
    let half = one / two;

    let (det_c, c_inv) = del_geo_core::mat3_array_of_array::det_inv(c);

    // [0.5*(J-1)*(J-1)]' -> (J-1) * J'ij -> (J-1) * J * Cij
    let dwdc = {
        let tmp0 = (det_c - one) * det_c;
        [
            tmp0 * c_inv[ISTDIM2IJ[0][0]][ISTDIM2IJ[0][1]],
            tmp0 * c_inv[ISTDIM2IJ[1][0]][ISTDIM2IJ[1][1]],
            tmp0 * c_inv[ISTDIM2IJ[2][0]][ISTDIM2IJ[2][1]],
            tmp0 * c_inv[ISTDIM2IJ[3][0]][ISTDIM2IJ[3][1]],
            tmp0 * c_inv[ISTDIM2IJ[4][0]][ISTDIM2IJ[4][1]],
            tmp0 * c_inv[ISTDIM2IJ[5][0]][ISTDIM2IJ[5][1]],
        ]
    };
    let ddwddc = {
        // Extracting independent components in the constitutive tensor
        let mut ddw_ddc = [[zero; 6]; 6];
        for (istdim, jstdim) in itertools::iproduct!(0..6, 0..6) {
            let idim = ISTDIM2IJ[istdim][0];
            let jdim = ISTDIM2IJ[istdim][1];
            let kdim = ISTDIM2IJ[jstdim][0];
            let ldim = ISTDIM2IJ[jstdim][1];
            /*
            // exact derivative
            // (J^2 - J) * Cij -> (2J - 1) * J'kl * Cij + (J^2-J) * C''ijkl
            // -> (2J - 1) * J * Cij * Ckl + (J^2-J) * Cik * Cjl
            let v0 = (two * det_c - one) * det_c *  c_inv[idim][jdim] * c_inv[kdim][ldim];
            let v1 = (one-det_c)*det_c * c_inv[idim][kdim] * c_inv[jdim][ldim];
            ddw_ddc[istdim][jstdim] = v0 + v1;
             */
            // symetrized derivative
            let v1 = (two * det_c - one) * det_c * c_inv[idim][jdim] * c_inv[kdim][ldim];
            let v2 = half * (one - det_c) * det_c * c_inv[idim][ldim] * c_inv[jdim][kdim];
            let v3 = half * (one - det_c) * det_c * c_inv[idim][kdim] * c_inv[jdim][ldim];
            ddw_ddc[istdim][jstdim] = v1 + v2 + v3;
        }
        ddw_ddc
    };
    (half * (det_c - one) * (det_c - one), dwdc, ddwddc)
}

pub fn tensor3_from_symmetric_vector_param<Real>(v: &[Real; 6]) -> [[Real; 3]; 3]
where
    Real: num_traits::Float,
{
    [[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]]
}

#[test]
pub fn test_hoge() {
    let cv0: [f64; 6] = [1., 0.9, 1.1, -0.1, -0.2, -0.3];
    let c0 = tensor3_from_symmetric_vector_param(&cv0);
    let (w0, dw0, ddw0) = wr_dwrdc_ddwrddc_energy_density_sqr_compression(&c0);
    let eps = 1.0e-6;
    for i_dim in 0..6 {
        let mut c1 = c0;
        c1[ISTDIM2IJ[i_dim][0]][ISTDIM2IJ[i_dim][1]] += eps;
        let (w1, _dw1, _ddw) = wr_dwrdc_ddwrddc_energy_density_sqr_compression(&c1);
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
        let (_w1, dw1, _ddw) = wr_dwrdc_ddwrddc_energy_density_sqr_compression(&c1);
        for j_dim in 0..6 {
            let v_num = (dw1[j_dim] - dw0[j_dim]) / eps;
            let v_ana = ddw0[i_dim][j_dim];
            assert!((v_num - v_ana).abs() < 5.0e-6);
        }
    }
}
