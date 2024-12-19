/**
 *
 * @param[out] C
 * @param[out] dCdp
 * @param[in] P undeformed triangle vertex positions
 * @param[in] p deformed triangle vertex positions
 */
pub fn wdw<Real>(p_ini: &[[Real; 2]; 3], p_def: &[[Real; 3]; 3]) -> ([Real; 3], [[[Real; 3]; 3]; 3])
where
    Real: num_traits::Float + std::ops::MulAssign,
{
    let zero = Real::zero();
    let one = Real::one();
    let two = one + one;
    let half = one / two;
    use del_geo_core::vec3;
    let gd0_x = [p_ini[1][0] - p_ini[0][0], p_ini[1][1] - p_ini[0][1], zero];
    let gd0_y = [p_ini[2][0] - p_ini[0][0], p_ini[2][1] - p_ini[0][1], zero];
    let gd0_z = vec3::cross(&gd0_x, &gd0_y);
    let area0 = vec3::norm(&gd0_z) * half;
    let gd0_z = vec3::scale(&gd0_z, one / (area0 * two));

    let gu0_x = vec3::cross(&gd0_y, &gd0_z);
    let gu0_x = vec3::normalize(&gu0_x);
    let gu0_y = vec3::cross(&gd0_z, &gd0_x);
    let gu0_y = vec3::normalize(&gu0_y);

    let gd1_x = [
        p_def[1][0] - p_def[0][0],
        p_def[1][1] - p_def[0][1],
        p_def[1][2] - p_def[0][2],
    ];
    let gd1_y = [
        p_def[2][0] - p_def[0][0],
        p_def[2][1] - p_def[0][1],
        p_def[2][2] - p_def[0][2],
    ];

    let glstrain = [
        // green lagrange strain (with engineer's notation)
        half * (vec3::dot(&gd1_x, &gd1_x) - vec3::dot(&gd0_x, &gd0_x)),
        half * (vec3::dot(&gd1_y, &gd1_y) - vec3::dot(&gd0_y, &gd0_y)),
        vec3::dot(&gd1_x, &gd1_y) - vec3::dot(&gd0_x, &gd0_y),
    ];

    let gugu0_xx = [
        gu0_x[0] * gu0_x[0],
        gu0_y[0] * gu0_y[0],
        gu0_x[0] * gu0_y[0],
    ];
    let gugu0_yy = [
        gu0_x[1] * gu0_x[1],
        gu0_y[1] * gu0_y[1],
        gu0_x[1] * gu0_y[1],
    ];
    let gugu0_xy = [
        two * gu0_x[0] * gu0_x[1],
        two * gu0_y[0] * gu0_y[1],
        gu0_x[0] * gu0_y[1] + gu0_x[1] * gu0_y[0],
    ];
    let mut w = [zero; 3];
    w[0] = glstrain[0] * gugu0_xx[0] + glstrain[1] * gugu0_xx[1] + glstrain[2] * gugu0_xx[2];
    w[1] = glstrain[0] * gugu0_yy[0] + glstrain[1] * gugu0_yy[1] + glstrain[2] * gugu0_yy[2];
    w[2] = glstrain[0] * gugu0_xy[0] + glstrain[1] * gugu0_xy[1] + glstrain[2] * gugu0_xy[2];

    let hoge = |d: &mut [Real; 3], a: Real, b: Real| {
        d[0] = a * gd1_x[0] + b * gd1_y[0];
        d[1] = a * gd1_x[1] + b * gd1_y[1];
        d[2] = a * gd1_x[2] + b * gd1_y[2];
    };

    let mut dw = [[[zero; 3]; 3]; 3];
    hoge(
        &mut dw[0][0],
        -(gugu0_xx[0] + gugu0_xx[2]),
        -(gugu0_xx[1] + gugu0_xx[2]),
    );
    hoge(&mut dw[0][1], gugu0_xx[0], gugu0_xx[2]);
    hoge(&mut dw[0][2], gugu0_xx[2], gugu0_xx[1]);
    hoge(
        &mut dw[1][0],
        -(gugu0_yy[0] + gugu0_yy[2]),
        -(gugu0_yy[1] + gugu0_yy[2]),
    );
    hoge(&mut dw[1][1], gugu0_yy[0], gugu0_yy[2]);
    hoge(&mut dw[1][2], gugu0_yy[2], gugu0_yy[1]);
    hoge(
        &mut dw[2][0],
        -(gugu0_xy[0] + gugu0_xy[2]),
        -(gugu0_xy[1] + gugu0_xy[2]),
    );
    hoge(&mut dw[2][1], gugu0_xy[0], gugu0_xy[2]);
    hoge(&mut dw[2][2], gugu0_xy[2], gugu0_xy[1]);
    (w, dw)

    /*
    let dC0dp0 = -(GuGu_xx[0] + GuGu_xx[2]) * gd0 - (GuGu_xx[1] + GuGu_xx[2]) * gd1;
    let dC0dp1 = GuGu_xx[0] * gd0 + GuGu_xx[2] * gd1;
    let dC0dp2 = GuGu_xx[1] * gd1 + GuGu_xx[2] * gd0;
    let dC1dp0 = -(GuGu_yy[0] + GuGu_yy[2]) * gd0 - (GuGu_yy[1] + GuGu_yy[2]) * gd1;
    let dC1dp1 = GuGu_yy[0] * gd0 + GuGu_yy[2] * gd1;
    let dC1dp2 = GuGu_yy[1] * gd1 + GuGu_yy[2] * gd0;
    let dC2dp0 = -(GuGu_xy[0] + GuGu_xy[2]) * gd0 - (GuGu_xy[1] + GuGu_xy[2]) * gd1;
    let dC2dp1 = GuGu_xy[0] * gd0 + GuGu_xy[2] * gd1;
    let dC2dp2 = GuGu_xy[1] * gd1 + GuGu_xy[2] * gd0;
     */
    /*
    dC0dp0.CopyTo(dCdp[0] + 0 * 3);
    dC0dp1.CopyTo(dCdp[0] + 1 * 3);
    dC0dp2.CopyTo(dCdp[0] + 2 * 3);
    dC1dp0.CopyTo(dCdp[1] + 0 * 3);
    dC1dp1.CopyTo(dCdp[1] + 1 * 3);
    dC1dp2.CopyTo(dCdp[1] + 2 * 3);
    dC2dp0.CopyTo(dCdp[2] + 0 * 3);
    dC2dp1.CopyTo(dCdp[2] + 1 * 3);
    dC2dp2.CopyTo(dCdp[2] + 2 * 3);
     */
}

#[test]
fn test() {
    let p_ini = [[0.1, 0.2], [0.2, 0.1], [0.3, 0.4]];
    let p0_def = [[0.31, 0.04, 0.38], [0.13, 0.13, 0.07], [0.03, 0.42, 0.35]];
    let (w0, dw) = wdw(&p_ini, &p0_def);
    let eps = 1.0e-6f64;
    for (i_no, i_dim) in itertools::iproduct!(0..3, 0..3) {
        let p1_def = {
            let mut p1_def = p0_def;
            p1_def[i_no][i_dim] += eps;
            p1_def
        };
        let (w1, _) = wdw(&p_ini, &p1_def);
        for j_dim in 0..3 {
            let v_ana = dw[j_dim][i_no][i_dim];
            let v_num = (w1[j_dim] - w0[j_dim]) / eps;
            // println!("{} {}", v_ana, v_num);
            assert!((v_ana - v_num).abs() < 1.0e-5, "{} {}", v_ana, v_num);
        }
    }
}

/// elastic potential energy (St.Venant-Kirchhoff material)
/// and its derivative and hessian w.r.t.
/// the deformed vertex position for a 3D triangle.
///
/// * `P` - un-deformed triangle vertex positions
/// * `p` - deformed triangle vertex positions
/// * `lambda` - Lame's 1st parameter
/// * `myu` - Lame's 2nd parameter
#[allow(non_snake_case)]
pub fn wdwddw_<T>(
    p0: [[T; 3]; 3],
    p1: [[T; 3]; 3],
    lambda: T,
    myu: T,
) -> (T, [[T; 3]; 3], [[[T; 9]; 3]; 3])
where
    T: num_traits::Float + std::ops::MulAssign + std::ops::AddAssign,
{
    use del_geo_core::tri3;
    use del_geo_core::vec3;

    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    let half = one / two;

    let (gd0, area0) = {
        let (gdz, area0) = tri3::unit_normal_area(&p0[0], &p0[1], &p0[2]);
        (
            [
                [
                    p0[1][0] - p0[0][0],
                    p0[1][1] - p0[0][1],
                    p0[1][2] - p0[0][2],
                ],
                [
                    p0[2][0] - p0[0][0],
                    p0[2][1] - p0[0][1],
                    p0[2][2] - p0[0][2],
                ],
                gdz,
            ],
            area0,
        )
    };

    let mut gu0: [[T; 3]; 2] = [[zero; 3]; 2]; // inverse of Gd
    {
        vec3::cross_mut(&mut gu0[0], &gd0[1], &gd0[2]);
        let invtmp1 = one / vec3::dot(&gu0[0], &gd0[0]);
        gu0[0][0] *= invtmp1;
        gu0[0][1] *= invtmp1;
        gu0[0][2] *= invtmp1;
        //
        vec3::cross_mut(&mut gu0[1], &gd0[2], &gd0[0]);
        let invtmp2 = one / vec3::dot(&gu0[1], &gd0[1]);
        gu0[1][0] *= invtmp2;
        gu0[1][1] *= invtmp2;
        gu0[1][2] *= invtmp2;
    }

    let gd: [[T; 3]; 2] = [
        // deformed edge vector
        [
            p1[1][0] - p1[0][0],
            p1[1][1] - p1[0][1],
            p1[1][2] - p1[0][2],
        ],
        [
            p1[2][0] - p1[0][0],
            p1[2][1] - p1[0][1],
            p1[2][2] - p1[0][2],
        ],
    ];

    let glstrain: [T; 3] = [
        // green lagrange strain (with engineer's notation)
        half * (vec3::dot(&gd[0], &gd[0]) - vec3::dot(&gd0[0], &gd0[0])),
        half * (vec3::dot(&gd[1], &gd[1]) - vec3::dot(&gd0[1], &gd0[1])),
        one * (vec3::dot(&gd[0], &gd[1]) - vec3::dot(&gd0[0], &gd0[1])),
    ];

    let gugu0: [T; 3] = [
        vec3::dot(&gu0[0], &gu0[0]),
        vec3::dot(&gu0[1], &gu0[1]),
        vec3::dot(&gu0[1], &gu0[0]),
    ];

    let elasticity_tensor: [[T; 3]; 3] = [
        // elasticity tensor
        [
            lambda * gugu0[0] * gugu0[0] + two * myu * (gugu0[0] * gugu0[0]),
            lambda * gugu0[0] * gugu0[1] + two * myu * (gugu0[2] * gugu0[2]),
            lambda * gugu0[0] * gugu0[2] + two * myu * (gugu0[0] * gugu0[2]),
        ],
        [
            lambda * gugu0[1] * gugu0[0] + two * myu * (gugu0[2] * gugu0[2]),
            lambda * gugu0[1] * gugu0[1] + two * myu * (gugu0[1] * gugu0[1]),
            lambda * gugu0[1] * gugu0[2] + two * myu * (gugu0[2] * gugu0[1]),
        ],
        [
            lambda * gugu0[2] * gugu0[0] + two * myu * (gugu0[0] * gugu0[2]),
            lambda * gugu0[2] * gugu0[1] + two * myu * (gugu0[2] * gugu0[1]),
            lambda * gugu0[2] * gugu0[2] + one * myu * (gugu0[0] * gugu0[1] + gugu0[2] * gugu0[2]),
        ],
    ];
    let spkstress: [T; 3] = [
        // 2nd Piola-Kirchhoff stress
        elasticity_tensor[0][0] * glstrain[0]
            + elasticity_tensor[0][1] * glstrain[1]
            + elasticity_tensor[0][2] * glstrain[2],
        elasticity_tensor[1][0] * glstrain[0]
            + elasticity_tensor[1][1] * glstrain[1]
            + elasticity_tensor[1][2] * glstrain[2],
        elasticity_tensor[2][0] * glstrain[0]
            + elasticity_tensor[2][1] * glstrain[1]
            + elasticity_tensor[2][2] * glstrain[2],
    ];

    // compute energy
    let w = half
        * area0
        * (glstrain[0] * spkstress[0] + glstrain[1] * spkstress[1] + glstrain[2] * spkstress[2]);

    // compute 1st derivative
    let dNdr: [[T; 2]; 3] = [[-one, -one], [one, zero], [zero, one]];
    let mut dw = [[T::zero(); 3]; 3];
    for (ino, idim) in itertools::iproduct!(0..3, 0..3) {
        dw[ino][idim] = area0
            * (spkstress[0] * gd[0][idim] * dNdr[ino][0]
                + spkstress[2] * gd[0][idim] * dNdr[ino][1]
                + spkstress[2] * gd[1][idim] * dNdr[ino][0]
                + spkstress[1] * gd[1][idim] * dNdr[ino][1]);
    }

    let spkstress: [T; 3] = [spkstress[0], spkstress[1], spkstress[2]];
    //MakePositiveDefinite_Sim22(S2, S3);

    // compute second derivative
    let mut ddw = [[[T::zero(); 9]; 3]; 3];
    for (ino, jno) in itertools::iproduct!(0..3, 0..3) {
        for (idim, jdim) in itertools::iproduct!(0..3, 0..3) {
            let mut dtmp0: T = zero;
            dtmp0 +=
                gd[0][idim] * dNdr[ino][0] * elasticity_tensor[0][0] * gd[0][jdim] * dNdr[jno][0];
            dtmp0 +=
                gd[0][idim] * dNdr[ino][0] * elasticity_tensor[0][1] * gd[1][jdim] * dNdr[jno][1];
            dtmp0 +=
                gd[0][idim] * dNdr[ino][0] * elasticity_tensor[0][2] * gd[0][jdim] * dNdr[jno][1];
            dtmp0 +=
                gd[0][idim] * dNdr[ino][0] * elasticity_tensor[0][2] * gd[1][jdim] * dNdr[jno][0];
            dtmp0 +=
                gd[1][idim] * dNdr[ino][1] * elasticity_tensor[1][0] * gd[0][jdim] * dNdr[jno][0];
            dtmp0 +=
                gd[1][idim] * dNdr[ino][1] * elasticity_tensor[1][1] * gd[1][jdim] * dNdr[jno][1];
            dtmp0 +=
                gd[1][idim] * dNdr[ino][1] * elasticity_tensor[1][2] * gd[0][jdim] * dNdr[jno][1];
            dtmp0 +=
                gd[1][idim] * dNdr[ino][1] * elasticity_tensor[1][2] * gd[1][jdim] * dNdr[jno][0];
            dtmp0 +=
                gd[0][idim] * dNdr[ino][1] * elasticity_tensor[2][0] * gd[0][jdim] * dNdr[jno][0];
            dtmp0 +=
                gd[0][idim] * dNdr[ino][1] * elasticity_tensor[2][1] * gd[1][jdim] * dNdr[jno][1];
            dtmp0 +=
                gd[0][idim] * dNdr[ino][1] * elasticity_tensor[2][2] * gd[0][jdim] * dNdr[jno][1];
            dtmp0 +=
                gd[0][idim] * dNdr[ino][1] * elasticity_tensor[2][2] * gd[1][jdim] * dNdr[jno][0];
            dtmp0 +=
                gd[1][idim] * dNdr[ino][0] * elasticity_tensor[2][0] * gd[0][jdim] * dNdr[jno][0];
            dtmp0 +=
                gd[1][idim] * dNdr[ino][0] * elasticity_tensor[2][1] * gd[1][jdim] * dNdr[jno][1];
            dtmp0 +=
                gd[1][idim] * dNdr[ino][0] * elasticity_tensor[2][2] * gd[0][jdim] * dNdr[jno][1];
            dtmp0 +=
                gd[1][idim] * dNdr[ino][0] * elasticity_tensor[2][2] * gd[1][jdim] * dNdr[jno][0];
            ddw[ino][jno][idim * 3 + jdim] = dtmp0 * area0;
        }
        let dtmp1 = area0
            * (spkstress[0] * dNdr[ino][0] * dNdr[jno][0]
                + spkstress[2] * dNdr[ino][0] * dNdr[jno][1]
                + spkstress[2] * dNdr[ino][1] * dNdr[jno][0]
                + spkstress[1] * dNdr[ino][1] * dNdr[jno][1]);
        ddw[ino][jno][0] += dtmp1;
        ddw[ino][jno][4] += dtmp1;
        ddw[ino][jno][8] += dtmp1;
    }
    (w, dw, ddw)
}

#[test]
fn test_wdwddw_cst() {
    let lambda = 1.3;
    let myu = 1.9;
    let test = |pos0, pos1| {
        let (w0, dw0, ddw0) = wdwddw_(pos0, pos1, lambda, myu);
        let eps = 1.0e-5_f64;
        for (ino, idim) in itertools::iproduct!(0..3, 0..3) {
            let mut pos1a = pos1.clone();
            pos1a[ino][idim] += eps;
            let (w1, dw1, _ddw1) = wdwddw_(pos0, pos1a, lambda, myu);
            let dw_numerical = (w1 - w0) / eps;
            let dw_analytical = dw0[ino][idim];
            assert!((dw_numerical - dw_analytical).abs() < 1.0e-4);
            for (jno, jdim) in itertools::iproduct!(0..3, 0..3) {
                let ddw_numerical = (dw1[jno][jdim] - dw0[jno][jdim]) / eps;
                let ddw_analytical = ddw0[jno][ino][jdim * 3 + idim];
                // dbg!(ddw_analytical, ddw_numerical);
                assert!((ddw_numerical - ddw_analytical).abs() < 1.0e-4);
            }
        }
    };
    test(
        [[1.2, 2.1, 3.4], [3.5, 5.2, 4.3], [3.4, 4.8, 2.4]],
        [[3.1, 2.2, 1.5], [4.3, 3.6, 2.0], [5.2, 4.5, 3.4]],
    );
    test(
        [[3.1, 5.0, 1.3], [2.0, 1.8, 3.4], [5.6, 2.4, 3.3]],
        [[2.5, 1.0, 3.2], [0.3, 4.6, 1.2], [3.2, 5.5, 1.4]],
    );
}
