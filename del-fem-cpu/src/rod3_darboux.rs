/// frame after small vertex movement of z0 and z-axis rotation
fn updated_rod_frame<T>(ux0: &[T; 3], z0: &[T; 3], dz: &[T; 3], dtheta: T) -> [[T; 3]; 3]
where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
    use del_geo_core::vec3::Vec3;
    /*
    let len_z0 = z0.norm();
    let invlen_z0 = T::one() / len_z0;
    let uz0 = z0.scale(invlen_z0);
    let uy0 = uz0.cross(ux0);
    // ux1 = exp{ skew(z) } * ux0 = { I + skew(z) } * ux0
    let ux1 = ux0.scale(dtheta.cos()).add(&uy0.scale(dtheta.sin()));
    // uy1 = exp{ skew(z) } * uy0 = { I + skew(z) } * uy0
    let uy1 = uy0.scale(dtheta.cos()).sub(&ux0.scale(dtheta.sin()));
    // let invlen_z1 = T::one() / (z0.add(&dz).norm());
    let r_mat = del_geo_core::mat3_col_major::from_axisangle_vec(&uz0.cross(dz).scale(invlen_z0));
    let ux2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &ux1);
    let uy2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &uy1);
    let uz2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &uz0);
    [ux2, uy2, uz2]
     */
    let len_z0 = z0.norm();
    let invlen_z0 = T::one() / len_z0;
    let uz0 = z0.scale(invlen_z0);
    let uz2 = z0.add(&dz).normalize();
    let uy0 = uz0.cross(ux0);
    let r_mat = del_geo_core::mat3_col_major::minimum_rotation_matrix(&uz0, &uz2);
    // ux1 = exp{ skew(z) } * ux0 = { I + skew(z) } * ux0
    let ux1 = ux0.scale(dtheta.cos()).add(&uy0.scale(dtheta.sin()));
    // uy1 = exp{ skew(z) } * uy0 = { I + skew(z) } * uy0
    let uy1 = uy0.scale(dtheta.cos()).sub(&ux0.scale(dtheta.sin()));
    let ux2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &ux1);
    let uy2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &uy1);
    [ux2, uy2, uz2]
}

/// # Argument
/// * `frame` - `frame[i][j]` is `frame[i]` is the `i`-th axis and `j`-th coordinate
/// # Return
/// * `dfdv` - differentiation of frame w.r.t vertex position (i.e., `frame[2] * len`) )
///   * `dfdv[i][j][k]` - differentiation of `frame[i][j]` w.r.t vertex position `v[k]`
fn rod_frame_gradient<T>(length: T, frame: &[[T; 3]; 3]) -> ([[[T; 3]; 3]; 3], [[T; 3]; 3])
where
    T: num_traits::Float,
{
    use del_geo_core::mat3_col_major;
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    let leninv = T::one() / length;
    let dfdv = [
        mat3_col_major::from_scaled_outer_product(-leninv, &frame[2], &frame[0])
            .to_mat3_array_of_array(),
        mat3_col_major::from_scaled_outer_product(-leninv, &frame[2], &frame[1])
            .to_mat3_array_of_array(),
        mat3_col_major::from_projection_onto_plane(&frame[2])
            .scale(leninv)
            .to_mat3_array_of_array(),
    ];
    use del_geo_core::vec3::Vec3;
    // [z^x, z^y, z^z]
    let dfdt = [frame[1], frame[0].scale(-T::one()), [T::zero(); 3]];
    (dfdv, dfdt)
}

/*
fn rod_frame_hessian(
    i_axis: usize,
    l01: f64,
    q: &[f64; 3],
    frm: &[[f64; 3]; 3],
) -> ([f64; 9], [f64; 3], f64) {
    use del_geo_core::mat3_col_major::{from_vec3_to_skew_mat, Mat3ColMajor};
    use del_geo_core::vec3::Vec3;
    let sz = from_vec3_to_skew_mat(&frm[2]);
    let se = from_vec3_to_skew_mat(&frm[i_axis]);
    let sq = from_vec3_to_skew_mat(&q);
    let se_sq = se.mult_mat_col_major(&sq);
    let se_sq_sym = se_sq.add(&se_sq.transpose());
    let ddv = sz
        .mult_mat_col_major(&se_sq_sym)
        .mult_mat_col_major(&sz)
        .scale(-0.5 / (l01 * l01));
    let ddt = q
        .mult_mat3_col_major(&sz)
        .mult_mat3_col_major(&sz)
        .dot(&frm[i_axis]);
    let dtdv = sz
        .mult_mat_col_major(&sq)
        .mult_mat_col_major(&sz)
        .mult_vec(&frm[i_axis])
        .scale(1.0 / l01);
    return (ddv, dtdv, ddt);
}
 */

#[test]
fn test_rod_frame_gradient_and_hessian() {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
    let eps = 1.0e-5f64;
    for _itr in 0..50 {
        use del_geo_core::vec3::Vec3;
        let q: [f64; 3] = [rng.random(), rng.random(), rng.random()];
        let len2 = rng.random::<f64>() + 0.1;
        let frm2 = {
            let ez =
                del_geo_core::sphere::sample_surface_uniform::<f64>(&[rng.random(), rng.random()]);
            let (ex, ey) = del_geo_core::vec3::basis_xy_from_basis_z(&ez);
            [ex, ey, ez]
        };
        //
        let du = [
            2.0 * rng.random::<f64>() - 1.0,
            2.0 * rng.random::<f64>() - 1.0,
            2.0 * rng.random::<f64>() - 1.0,
            //0., 0., 0.,
        ];
        let dt = 2.0 * rng.random::<f64>() - 1.0;
        let (dw2dv, dw2dt) = {
            let (dfdv, dfdt) = rod_frame_gradient(len2, &frm2);
            let dwdv = [
                del_geo_core::vec3::mult_mat3_array_of_array(&q, &dfdv[0]),
                del_geo_core::vec3::mult_mat3_array_of_array(&q, &dfdv[1]),
                del_geo_core::vec3::mult_mat3_array_of_array(&q, &dfdv[2]),
            ];
            let dwdt = [
                del_geo_core::vec3::dot(&q, &dfdt[0]),
                del_geo_core::vec3::dot(&q, &dfdt[1]),
                del_geo_core::vec3::dot(&q, &dfdt[2]),
            ];
            (dwdv, dwdt)
        };
        // dbg!(dw3dv);
        let frm4 = updated_rod_frame(&frm2[0], &frm2[2].scale(len2), &du.scale(eps), dt * eps);
        let w4 = [q.dot(&frm4[0]), q.dot(&frm4[1]), q.dot(&frm4[2])];
        let frm0 = updated_rod_frame(&frm2[0], &frm2[2].scale(len2), &du.scale(-eps), -dt * eps);
        let w0 = [q.dot(&frm0[0]), q.dot(&frm0[1]), q.dot(&frm0[2])];
        for i in 0..3 {
            let a = (w4[i] - w0[i]) * 0.5 / eps;
            let b = dw2dv[i].dot(&du) + dw2dt[i] * dt;
            let err = (a - b).abs() / (a.abs() + b.abs() + 1.0e-20);
            assert!(err < 4.0e-3, "{} {} {} {}", i, a, b, err);
        }
        /*
        let w2 = [q.dot(&frm2[0]), q.dot(&frm2[1]), q.dot(&frm2[2])];
        for i_axis in 0..3 {
            let (ddw2ddv, ddwdtdv, ddwddt) = rod_frame_hessian(i_axis, len2, &q, &frm2);
            let a = (w0[i_axis] + w4[i_axis] - 2.0 * w2[i_axis]) / (eps * eps);
            let b = del_geo_core::mat3_col_major::mult_vec(&ddw2ddv, &du).dot(&du)
                + ddwddt * dt * dt
                + 2.0 * ddwdtdv.dot(&du) * dt;
            let err = (a - b).abs() / (a.abs() + b.abs() + 3.0e-3);
            assert!(err < 3.0e-3, "{} {} {}  {}", a, b, (a - b).abs(), err);
        }
         */
    }
}

// above gradient and hessian of frame
// ---------------------------------------

// add derivative of dot( Frm0[i], Frm1[j] ) with respect to the 3 points and 2 rotations
// of the rod element
pub fn add_gradient_of_dot_frame_axis<T>(
    dvdp: &mut [[T; 3]; 3],
    dvdt: &mut [T; 2],
    c: T,
    i0_axis: usize,
    frma: &[[T; 3]; 3],
    dfadp: &[[[T; 3]; 3]; 3],
    dfadt: &[[T; 3]; 3],
    i1_axis: usize,
    frmb: &[[T; 3]; 3],
    dfbdp: &[[[T; 3]; 3]; 3],
    dfbdt: &[[T; 3]; 3],
) where
    T: num_traits::Float,
{
    use del_geo_core::vec3::Vec3;
    dvdt[0] = dvdt[0] + c * frmb[i1_axis].dot(&dfadt[i0_axis]);
    dvdt[1] = dvdt[1] + c * frma[i0_axis].dot(&dfbdt[i1_axis]);
    {
        let tmp0 = frmb[i1_axis]
            .mult_mat3_array_of_array(&dfadp[i0_axis])
            .scale(c);
        dvdp[0].sub_in_place(&tmp0);
        dvdp[1].add_in_place(&tmp0);
    }
    {
        let tmp0 = frma[i0_axis]
            .mult_mat3_array_of_array(&dfbdp[i1_axis])
            .scale(c);
        dvdp[1].sub_in_place(&tmp0);
        dvdp[2].add_in_place(&tmp0);
    }
}

fn darboux_rod<T>(p: &[[T; 3]; 3], x: &[[T; 3]; 2]) -> [T; 3]
where
    T: num_traits::Float,
{
    let one = T::one();
    //assert(fabs(S[0].norm() - 1.0) < 1.0e-5);
    //assert(fabs(S[0].dot((P[1] - P[0]).normalized())) < 1.0e-5);
    //assert(fabs(S[1].norm() - 1.0) < 1.0e-5);
    //assert(fabs(S[1].dot((P[2] - P[1]).normalized())) < 1.0e-5);

    use del_geo_core::vec3::Vec3;
    let (frma, _lena) = {
        let z = p[1].sub(&p[0]);
        let len = z.norm();
        let uz = z.normalize();
        let uy = uz.cross(&x[0]);
        ([x[0], uy, uz], len)
    };
    let (frmb, _lenb) = {
        let z = p[2].sub(&p[1]);
        let len = z.norm();
        let uz = z.normalize();
        let uy = uz.cross(&x[1]);
        ([x[1], uy, uz], len)
    };
    let s = one + frma[0].dot(&frmb[0]) + frma[1].dot(&frmb[1]) + frma[2].dot(&frmb[2]);
    let u = [
        frma[1].dot(&frmb[2]) - frma[2].dot(&frmb[1]),
        frma[2].dot(&frmb[0]) - frma[0].dot(&frmb[2]),
        frma[0].dot(&frmb[1]) - frma[1].dot(&frmb[0]),
    ];
    [u[0] / s, u[1] / s, u[2] / s]
}

/// Darboux vector in the reference configuration and its gradient
pub fn cdc_rod_darboux<T>(
    p: &[[T; 3]; 3],
    x: &[[T; 3]; 2],
) -> ([T; 3], [[[T; 3]; 3]; 3], [[T; 2]; 3])
where
    T: num_traits::Float,
{
    let zero = T::zero();
    let one = T::one();
    use del_geo_core::vec3::Vec3;
    let (frma, lena) = {
        let z = p[1].sub(&p[0]);
        let len = z.norm();
        let uz = z.normalize();
        let uy = uz.cross(&x[0]);
        ([x[0], uy, uz], len)
    };
    let (frmb, lenb) = {
        let z = p[2].sub(&p[1]);
        let len = z.norm();
        let uz = z.normalize();
        let uy = uz.cross(&x[1]);
        ([x[1], uy, uz], len)
    };
    //
    let (dfadp, dfadt) = rod_frame_gradient(lena, &frma);
    let (dfbdp, dfbdt) = rod_frame_gradient(lenb, &frmb);
    let s = T::one() + frma[0].dot(&frmb[0]) + frma[1].dot(&frmb[1]) + frma[2].dot(&frmb[2]);
    let (dsdp, dsdt) = {
        // making derivative of Y
        let mut dsdp = [[zero; 3]; 3];
        let mut dsdt = [zero; 2];
        add_gradient_of_dot_frame_axis(
            &mut dsdp, &mut dsdt, one, 0, &frma, &dfadp, &dfadt, 0, &frmb, &dfbdp, &dfbdt,
        );
        add_gradient_of_dot_frame_axis(
            &mut dsdp, &mut dsdt, one, 1, &frma, &dfadp, &dfadt, 1, &frmb, &dfbdp, &dfbdt,
        );
        add_gradient_of_dot_frame_axis(
            &mut dsdp, &mut dsdt, one, 2, &frma, &dfadp, &dfadt, 2, &frmb, &dfbdp, &dfbdt,
        );
        (dsdp, dsdt)
    };
    let mut c = [zero; 3];
    let mut dcdp = [[[zero; 3]; 3]; 3];
    let mut dcdt = [[zero; 2]; 3];
    for iaxis in 0..3 {
        let jaxis = (iaxis + 1) % 3;
        let kaxis = (iaxis + 2) % 3;
        let u = frma[jaxis].dot(&frmb[kaxis]) - frma[kaxis].dot(&frmb[jaxis]);
        let mut dudp = [[zero; 3]; 3];
        let mut dudt = [zero, zero];
        {
            add_gradient_of_dot_frame_axis(
                &mut dudp, &mut dudt, one, jaxis, &frma, &dfadp, &dfadt, kaxis, &frmb, &dfbdp,
                &dfbdt,
            );
            add_gradient_of_dot_frame_axis(
                &mut dudp, &mut dudt, -one, kaxis, &frma, &dfadp, &dfadt, jaxis, &frmb, &dfbdp,
                &dfbdt,
            );
        }
        c[iaxis] = u / s;
        {
            let t0 = one / s;
            let t1 = -u / (s * s);
            dcdp[iaxis][0] = dudp[0].scale(t0).add(&dsdp[0].scale(t1));
            dcdp[iaxis][1] = dudp[1].scale(t0).add(&dsdp[1].scale(t1));
            dcdp[iaxis][2] = dudp[2].scale(t0).add(&dsdp[2].scale(t1));
            dcdt[iaxis][0] = dudt[0] * t0 + dsdt[0] * t1;
            dcdt[iaxis][1] = dudt[1] * t0 + dsdt[1] * t1;
        }
    }
    (c, dcdp, dcdt)
}

#[test]
fn test_dot_rod_frame_gradient_and_hessian() {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
    let eps = 1.0e-4;
    for _itr in 0..100 {
        use del_geo_core::vec3::Vec3;
        let p2: [[f64; 3]; 3] = [
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
        ];
        {
            // reject
            let v10 = p2[1].sub(&p2[0]);
            let v12 = p2[2].sub(&p2[1]);
            if v10.norm() < 0.01 {
                continue;
            }
            if v12.norm() < 0.01 {
                continue;
            }
            if del_geo_core::tri3::angle(&p2[0], &p2[1], &p2[2]) < 0.3 {
                continue;
            }
        }
        let x2 = {
            let x0 = [rng.random(), rng.random(), rng.random()];
            let v10 = p2[1].sub(&p2[0]);
            let x0 = del_geo_core::vec3::orthogonalize(&v10, &x0).normalize();
            let x1 = [rng.random(), rng.random(), rng.random()];
            let v12 = p2[2].sub(&p2[1]);
            let x1 = del_geo_core::vec3::orthogonalize(&v12, &x1).normalize();
            [x0, x1]
        };
        let (_c2, dc2dp, dc2dt) = cdc_rod_darboux(&p2, &x2);
        let dp: [[f64; 3]; 3] = [
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
        ];
        let dt: [f64; 2] = [
            2.0 * rng.random::<f64>() - 1.0,
            2.0 * rng.random::<f64>() - 1.0,
        ];
        let p4 = [
            p2[0].add(&dp[0].scale(eps)),
            p2[1].add(&dp[1].scale(eps)),
            p2[2].add(&dp[2].scale(eps)),
        ];
        let x4 = {
            let frma = updated_rod_frame(
                &x2[0],
                &p2[1].sub(&p2[0]),
                &dp[1].sub(&dp[0]).scale(eps),
                dt[0] * eps,
            );
            let frmb = updated_rod_frame(
                &x2[1],
                &p2[2].sub(&p2[1]),
                &dp[2].sub(&dp[1]).scale(eps),
                dt[1] * eps,
            );
            [frma[0], frmb[0]]
        };
        let (c4, _dc4dp, _dc4dt) = cdc_rod_darboux(&p4, &x4);
        //
        let p0 = [
            p2[0].add(&dp[0].scale(-eps)),
            p2[1].add(&dp[1].scale(-eps)),
            p2[2].add(&dp[2].scale(-eps)),
        ];
        let x0 = {
            let frma = updated_rod_frame(
                &x2[0],
                &p2[1].sub(&p2[0]),
                &dp[1].sub(&dp[0]).scale(-eps),
                dt[0] * -eps,
            );
            let frmb = updated_rod_frame(
                &x2[1],
                &p2[2].sub(&p2[1]),
                &dp[2].sub(&dp[1]).scale(-eps),
                dt[1] * -eps,
            );
            [frma[0], frmb[0]]
        };
        let (c0, _dc0dp, _dc0dt) = cdc_rod_darboux(&p0, &x0);
        for iaxis in 0..3 {
            let v_num = (c4[iaxis] - c0[iaxis]) * 0.5 / eps;
            let v_ana = dc2dp[iaxis][0].dot(&dp[0])
                + dc2dp[iaxis][1].dot(&dp[1])
                + dc2dp[iaxis][2].dot(&dp[2])
                + dc2dt[iaxis][0] * dt[0]
                + dc2dt[iaxis][1] * dt[1];
            // println!("{} {} {}", iaxis, v_num, v_ana);
            let err = (v_num - v_ana).abs() / (v_num.abs() + v_ana.abs() + 1.0);
            assert!(err < 5.0e-5, "{}", err);
        }
    }
}

fn wdwdwdw_darboux_rod_hair_approx_hessian<T>(
    p: &[[T; 3]; 3],
    x: &[[T; 3]; 2],
    stiff_bendtwist: &[T; 3],
    darboux0: &[T; 3],
) -> (T, [[T; 4]; 3], [[[T; 16]; 3]; 3])
where
    T: num_traits::Float,
{
    let zero = T::zero();
    let one = T::one();
    let half = one / (one + one);
    use del_geo_core::vec3::Vec3;
    let (c, dcdp, dcdt) = cdc_rod_darboux(p, x);
    let r = [c[0] - darboux0[0], c[1] - darboux0[1], c[2] - darboux0[2]];
    let w = half
        * (stiff_bendtwist[0] * r[0] * r[0]
            + stiff_bendtwist[1] * r[1] * r[1]
            + stiff_bendtwist[2] * r[2] * r[2]);
    let dw = {
        let mut dw = [[zero; 4]; 3];
        for ino in 0..3 {
            let t0 = dcdp[0][ino].scale(stiff_bendtwist[0] * r[0]);
            let t1 = dcdp[1][ino].scale(stiff_bendtwist[1] * r[1]);
            let t2 = dcdp[2][ino].scale(stiff_bendtwist[2] * r[2]);
            let t = del_geo_core::vec3::add_three(&t0, &t1, &t2);
            dw[ino][0] = t[0];
            dw[ino][1] = t[1];
            dw[ino][2] = t[2];
        }
        for ino in 0..2 {
            dw[ino][3] = stiff_bendtwist[0] * r[0] * dcdt[0][ino]
                + stiff_bendtwist[1] * r[1] * dcdt[1][ino]
                + stiff_bendtwist[2] * r[2] * dcdt[2][ino];
        }
        dw[2][3] = zero;
        dw
    };
    let ddw = {
        let mut ddw = [[[zero; 16]; 3]; 3];
        for ino in 0..3 {
            for jno in 0..3 {
                let m0 = del_geo_core::mat3_col_major::from_scaled_outer_product(
                    stiff_bendtwist[0],
                    &dcdp[0][ino],
                    &dcdp[0][jno],
                );
                let m1 = del_geo_core::mat3_col_major::from_scaled_outer_product(
                    stiff_bendtwist[1],
                    &dcdp[1][ino],
                    &dcdp[1][jno],
                );
                let m2 = del_geo_core::mat3_col_major::from_scaled_outer_product(
                    stiff_bendtwist[2],
                    &dcdp[2][ino],
                    &dcdp[2][jno],
                );
                let m = del_geo_core::mat3_col_major::add_three(&m0, &m1, &m2);
                // this put one at `ddw[ino][jno][4*3+3]`
                ddw[ino][jno] =
                    del_geo_core::mat4_col_major::from_mat3_col_major_adding_w(&m, zero);
                ddw[ino][jno][15] = zero;
            }
        }
        for ino in 0..3 {
            // displacement node
            for jno in 0..2 {
                // rotation node
                let v0 = dcdp[0][ino].scale(stiff_bendtwist[0] * dcdt[0][jno]);
                let v1 = dcdp[1][ino].scale(stiff_bendtwist[1] * dcdt[1][jno]);
                let v2 = dcdp[2][ino].scale(stiff_bendtwist[2] * dcdt[2][jno]);
                let v = del_geo_core::vec3::add_three(&v0, &v1, &v2);
                ddw[ino][jno][12] = v[0];
                ddw[ino][jno][13] = v[1];
                ddw[ino][jno][14] = v[2];
                ddw[jno][ino][3] = v[0];
                ddw[jno][ino][7] = v[1];
                ddw[jno][ino][11] = v[2]
            }
        }
        for ino in 0..2 {
            for jno in 0..2 {
                ddw[ino][jno][15] = stiff_bendtwist[0] * dcdt[0][ino] * dcdt[0][jno]
                    + stiff_bendtwist[1] * dcdt[1][ino] * dcdt[1][jno]
                    + stiff_bendtwist[2] * dcdt[2][ino] * dcdt[2][jno];
            }
        }
        ddw
    };
    (w, dw, ddw)
}

#[test]
fn test_darboux_rod_hari_approx_hessian() {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
    let eps = 1.0e-4;
    for _itr in 0..50 {
        use del_geo_core::vec3::Vec3;
        let p2: [[f64; 3]; 3] = [
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
            del_geo_core::ndc::sample_inside_uniformly(&mut rng),
        ];
        {
            // reject
            let v10 = p2[1].sub(&p2[0]);
            let v12 = p2[2].sub(&p2[1]);
            if v10.norm() < 0.01 {
                continue;
            }
            if v12.norm() < 0.01 {
                continue;
            }
            if del_geo_core::tri3::angle(&p2[0], &p2[1], &p2[2]) < 0.3 {
                continue;
            }
        }
        let x2 = {
            let x0 = [rng.random(), rng.random(), rng.random()];
            let v10 = p2[1].sub(&p2[0]);
            let x0 = del_geo_core::vec3::orthogonalize(&v10, &x0).normalize();
            let x1 = [rng.random(), rng.random(), rng.random()];
            let v12 = p2[2].sub(&p2[1]);
            let x1 = del_geo_core::vec3::orthogonalize(&v12, &x1).normalize();
            [x0, x1]
        };
        //let p2: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]];
        //let x2: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let stiff_bendtwist = [1.0, 1.0, 1.0];
        let darboux_ini = [0.1, 0.1, 0.1];
        let (_w2, dw2dpt, ddw2ddpt) =
            wdwdwdw_darboux_rod_hair_approx_hessian(&p2, &x2, &stiff_bendtwist, &darboux_ini);
        {
            let (c, dcdp, dcdt) = cdc_rod_darboux(&p2, &x2);
            let dpt: [[f64; 4]; 3] = [
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ],
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ],
                [
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                    rng.random::<f64>(),
                ],
            ];
            let dc = {
                let mut dc = [0.0; 3];
                for ino in 0..3 {
                    for k in 0..3 {
                        dc[0] += dcdp[0][ino][k] * dpt[ino][k];
                        dc[1] += dcdp[1][ino][k] * dpt[ino][k];
                        dc[2] += dcdp[2][ino][k] * dpt[ino][k];
                    }
                }
                dc[0] += dcdt[0][0] * dpt[0][3] + dcdt[0][1] * dpt[1][3];
                dc[1] += dcdt[1][0] * dpt[0][3] + dcdt[1][1] * dpt[1][3];
                dc[2] += dcdt[2][0] * dpt[0][3] + dcdt[2][1] * dpt[1][3];
                dc
            };
            let dw0 = {
                let mut dw = 0.;
                for ino in 0..3 {
                    for k in 0..4 {
                        dw += dw2dpt[ino][k] * dpt[ino][k];
                    }
                }
                dw
            };
            let dw1 = {
                (c[0] - darboux_ini[0]) * stiff_bendtwist[0] * dc[0]
                    + (c[1] - darboux_ini[1]) * stiff_bendtwist[1] * dc[1]
                    + (c[2] - darboux_ini[2]) * stiff_bendtwist[2] * dc[2]
            };
            assert!((dw0 - dw1).abs() < 1.0e-11, "{}", (dw0 - dw1).abs());
            let dwdw0 = {
                let mut dwdw = 0.;
                for ino in 0..3 {
                    for jno in 0..3 {
                        let a = &ddw2ddpt[ino][jno];
                        let b = del_geo_core::mat4_col_major::mult_vec(&a, &dpt[jno]);
                        let c = del_geo_core::vecn::dot::<f64, 4>(&b, &dpt[ino]);
                        dwdw += c;
                    }
                }
                dwdw
            };
            let dwdw1 = dc[0] * dc[0] * stiff_bendtwist[0]
                + dc[1] * dc[1] * stiff_bendtwist[1]
                + dc[2] * dc[2] * stiff_bendtwist[2];
            assert!(
                (dwdw0 - dwdw1).abs() < 1.0e-11,
                "{dwdw0}, {dwdw1}, {}",
                (dwdw0 - dwdw1).abs()
            );
        }
        {
            // check gradient
            let dp: [[f64; 3]; 3] = [
                del_geo_core::ndc::sample_inside_uniformly(&mut rng),
                del_geo_core::ndc::sample_inside_uniformly(&mut rng),
                del_geo_core::ndc::sample_inside_uniformly(&mut rng),
            ];
            let dt: [f64; 2] = [
                2.0 * rng.random::<f64>() - 1.0,
                2.0 * rng.random::<f64>() - 1.0,
            ];
            let p4 = [
                p2[0].add(&dp[0].scale(eps)),
                p2[1].add(&dp[1].scale(eps)),
                p2[2].add(&dp[2].scale(eps)),
            ];
            let x4 = {
                let frma = updated_rod_frame(
                    &x2[0],
                    &p2[1].sub(&p2[0]),
                    &dp[1].sub(&dp[0]).scale(eps),
                    dt[0] * eps,
                );
                let frmb = updated_rod_frame(
                    &x2[1],
                    &p2[2].sub(&p2[1]),
                    &dp[2].sub(&dp[1]).scale(eps),
                    dt[1] * eps,
                );
                [frma[0], frmb[0]]
            };
            let (w4, _dw4dpt, _ddw4ddpt) =
                wdwdwdw_darboux_rod_hair_approx_hessian(&p4, &x4, &stiff_bendtwist, &darboux_ini);
            //
            let p0 = [
                p2[0].add(&dp[0].scale(-eps)),
                p2[1].add(&dp[1].scale(-eps)),
                p2[2].add(&dp[2].scale(-eps)),
            ];
            let x0 = {
                let frma = updated_rod_frame(
                    &x2[0],
                    &p2[1].sub(&p2[0]),
                    &dp[1].sub(&dp[0]).scale(-eps),
                    dt[0] * -eps,
                );
                let frmb = updated_rod_frame(
                    &x2[1],
                    &p2[2].sub(&p2[1]),
                    &dp[2].sub(&dp[1]).scale(-eps),
                    dt[1] * -eps,
                );
                [frma[0], frmb[0]]
            };
            let (w0, _dw0dpt, _ddw0ddpt) =
                wdwdwdw_darboux_rod_hair_approx_hessian(&p0, &x0, &stiff_bendtwist, &darboux_ini);
            //
            let dpt = [
                [dp[0][0], dp[0][1], dp[0][2], dt[0]],
                [dp[1][0], dp[1][1], dp[1][2], dt[1]],
                [dp[2][0], dp[2][1], dp[2][2], 0.0],
            ];
            let dw_num = (w4 - w0) * 0.5 / eps;
            let dw_ana = {
                let mut dw = 0.0;
                for i in 0..3 {
                    for j in 0..4 {
                        dw += dw2dpt[i][j] * dpt[i][j];
                    }
                }
                dw
            };
            // dw2dpt^2 ddw2ddpt の比較
            let err = (dw_num - dw_ana).abs() / (dw_num.abs() + dw_ana.abs() + 1.0);
            //dbg!(dw_ana, dw_num, err);
            assert!(err < 2.0e-4, "{}", err);
        }
    }
}

pub fn make_config_darboux_simple() -> (Vec<f64>, Vec<f64>) {
    let num_vtx = 30;
    let elen = 0.2;
    let vtx2xyz = {
        let mut vtx2xyz = vec![];
        for i_vtx in 0..num_vtx {
            let pos = [elen * (i_vtx) as f64, 0.0, 0.0];
            vtx2xyz.push(pos[0]);
            vtx2xyz.push(pos[1]);
            vtx2xyz.push(pos[2]);
        }
        vtx2xyz
    };
    let vtx2framex = {
        let mut vtx2framex = vec![];
        for _i_vtx in 0..num_vtx {
            vtx2framex.push(0.);
            vtx2framex.push(0.);
            vtx2framex.push(1.);
        }
        vtx2framex
    };
    (vtx2xyz, vtx2framex)
}

pub fn orthonormalize_framex_for_hair<T>(vtx2framex: &mut [T], vtx2xyz: &[T])
where
    T: num_traits::Float,
{
    use del_geo_core::vec3::Vec3;
    use slice_of_array::SliceNestExt;
    let vtx2framex = vtx2framex.nest_mut();
    let vtx2xyz: &[[T; 3]] = vtx2xyz.nest();
    for i0_vtx in 0..vtx2xyz.len() - 1 {
        let i1_vtx = i0_vtx + 1;
        let p0 = &vtx2xyz[i0_vtx];
        let p1 = &vtx2xyz[i1_vtx];
        let v01 = p1.sub(p0);
        vtx2framex[i0_vtx] =
            del_geo_core::vec3::orthogonalize(&v01, &vtx2framex[i0_vtx]).normalize();
    }
}

pub fn wdwddw_hair_system<T>(
    w: &mut T,
    dw: &mut [[T; 4]],
    mut ddw: crate::sparse_square::MatrixRefMut<T, 16>,
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    stiff_length: T,
    stiff_bendtwist: &[T; 3],
    vtx2framex_ini: &[T],
    vtx2framex_def: &[T],
    diagonal_damp: T,
) where
    T: num_traits::Float,
{
    let zero = T::zero();
    *w = zero;
    dw.fill([zero; 4]);
    ddw.set_zero();
    let num_vtx = vtx2xyz_ini.len() / 3;
    let mut col2idx = vec![usize::MAX; num_vtx];

    for i0_vtx in 0..num_vtx - 1 {
        let i1_vtx = i0_vtx + 1;
        let length_ini = {
            let p0 = del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_ini, i0_vtx);
            let p1 = del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_ini, i1_vtx);
            del_geo_core::edge3::length(&p0, &p1)
        };
        let p1 = [
            del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_def, i0_vtx),
            del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_def, i1_vtx),
        ];
        let (w_e, dw_e, ddw_e) =
            crate::spring3::wdwddw_squared_length_difference(stiff_length, &p1, length_ini);
        *w = (*w) + w_e;
        {
            use del_geo_core::vec4::Vec4;
            dw[i0_vtx].add_in_place(&del_geo_core::vec3::to_vec4_adding_w(&dw_e[0], zero));
            dw[i1_vtx].add_in_place(&del_geo_core::vec3::to_vec4_adding_w(&dw_e[1], zero));
        }
        let ddw_e = [
            [
                del_geo_core::mat4_col_major::from_mat3_col_major_adding_w(&ddw_e[0][0], zero),
                del_geo_core::mat4_col_major::from_mat3_col_major_adding_w(&ddw_e[0][1], zero),
            ],
            [
                del_geo_core::mat4_col_major::from_mat3_col_major_adding_w(&ddw_e[1][0], zero),
                del_geo_core::mat4_col_major::from_mat3_col_major_adding_w(&ddw_e[1][1], zero),
            ],
        ];
        ddw.merge_for_array_blk(&ddw_e, &[i0_vtx, i1_vtx], &mut col2idx);
    }
    for i0_vtx in 0..num_vtx - 2 {
        let i1_vtx = i0_vtx + 1;
        let i2_vtx = i0_vtx + 2;
        let p0 = [
            del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_ini, i0_vtx),
            del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_ini, i1_vtx),
            del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_ini, i2_vtx),
        ];
        let x0 = [
            del_msh_cpu::vtx2xyz::to_array3(&vtx2framex_ini, i0_vtx),
            del_msh_cpu::vtx2xyz::to_array3(&vtx2framex_ini, i1_vtx),
        ];
        let darboux0 = darboux_rod(&p0, &x0);
        let p1 = [
            del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_def, i0_vtx),
            del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_def, i1_vtx),
            del_msh_cpu::vtx2xyz::to_array3(&vtx2xyz_def, i2_vtx),
        ];
        let x1 = [
            del_msh_cpu::vtx2xyz::to_array3(&vtx2framex_def, i0_vtx),
            del_msh_cpu::vtx2xyz::to_array3(&vtx2framex_def, i1_vtx),
        ];
        let (w_e, dw_e, ddw_e) =
            wdwdwdw_darboux_rod_hair_approx_hessian(&p1, &x1, &stiff_bendtwist, &darboux0);
        *w = (*w) + w_e;
        {
            use del_geo_core::vec4::Vec4;
            dw[i0_vtx].add_in_place(&dw_e[0]);
            dw[i1_vtx].add_in_place(&dw_e[1]);
            dw[i2_vtx].add_in_place(&dw_e[2]);
        }
        ddw.merge_for_array_blk(&ddw_e, &[i0_vtx, i1_vtx, i2_vtx], &mut col2idx);
    }
    for i_vtx in 0..num_vtx {
        ddw.row2val[i_vtx][0] = ddw.row2val[i_vtx][0] + diagonal_damp;
        ddw.row2val[i_vtx][5] = ddw.row2val[i_vtx][5] + diagonal_damp;
        ddw.row2val[i_vtx][10] = ddw.row2val[i_vtx][10] + diagonal_damp;
        // ddw.row2val[i_vtx][15] += eps;
    }
}

pub fn update_solution_hair<T>(
    vtx2xyz: &mut [T],
    vtx2framex: &mut [T],
    vec_x: &[[T; 4]],
    vtx2isfix: &[[i32; 4]],
) where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz.len() / 3;
    for i0_vtx in 0..num_vtx - 1 {
        if vtx2isfix[i0_vtx][3] != 0 {
            continue;
        }
        let i1_vtx = i0_vtx + 1;
        let p0 = arrayref::array_ref![&vtx2xyz, i0_vtx * 3, 3];
        let p1 = arrayref::array_ref![&vtx2xyz, i1_vtx * 3, 3];
        let z = p1.sub(&p0);
        let dz = [
            vec_x[i0_vtx][0] - vec_x[i1_vtx][0],
            vec_x[i0_vtx][1] - vec_x[i1_vtx][1],
            vec_x[i0_vtx][2] - vec_x[i1_vtx][2],
        ];
        let dtheta = -vec_x[i0_vtx][3];
        let framex = arrayref::array_ref![&vtx2framex, i0_vtx * 3, 3];
        // let tmp0 = z.dot(framex);
        let frm = updated_rod_frame(framex, &z, &dz, dtheta);
        // let tmp1 = z.add(&dz).normalize().dot(&frm[0]);
        // println!("{} {} {}", i0_vtx, tmp0, tmp1);
        vtx2framex[i0_vtx * 3] = frm[0][0];
        vtx2framex[i0_vtx * 3 + 1] = frm[0][1];
        vtx2framex[i0_vtx * 3 + 2] = frm[0][2];
    }
    for i_vtx in 0..num_vtx {
        for i_dim in 0..3 {
            if vtx2isfix[i_vtx][i_dim] != 0 {
                continue;
            };
            vtx2xyz[i_vtx * 3 + i_dim] = vtx2xyz[i_vtx * 3 + i_dim] - vec_x[i_vtx][i_dim];
        }
    }
}

pub fn initialize_with_perturbation<T, Rng>(
    vtx2xyz_def: &mut [T],
    vtx2framex_def: &mut [T],
    vtx2xyz_ini: &[T],
    vtx2framex_ini: &[T],
    vtx2isfix: &[[i32; 4]],
    pos_mag: T,
    framex_mag: T,
    mut rng: Rng,
) where
    T: num_traits::Float,
    Rng: rand::Rng,
    rand::distr::StandardUniform: rand::prelude::Distribution<T>,
{
    let one = T::one();
    let two = one + one;

    let num_vtx = vtx2xyz_ini.len() / 3;
    vtx2xyz_def.copy_from_slice(&vtx2xyz_ini);
    vtx2framex_def.copy_from_slice(&vtx2framex_ini);
    for i_vtx in 0..num_vtx {
        for i_dim in 0..3 {
            if vtx2isfix[i_vtx][i_dim] == 0 {
                vtx2xyz_def[i_vtx * 3 + i_dim] =
                    vtx2xyz_def[i_vtx * 3 + i_dim] + (two * rng.random::<T>() - one) * pos_mag;
            }
        }
        if vtx2isfix[i_vtx][3] == 0 {
            let r: [T; 3] = std::array::from_fn(|_v| (two * rng.random::<T>() - one) * framex_mag);
            vtx2framex_def[i_vtx * 3] = vtx2framex_def[i_vtx * 3] + r[0];
            vtx2framex_def[i_vtx * 3 + 1] = vtx2framex_def[i_vtx * 3 + 1] + r[1];
            vtx2framex_def[i_vtx * 3 + 2] = vtx2framex_def[i_vtx * 3 + 2] + r[2];
        }
    }
    orthonormalize_framex_for_hair(vtx2framex_def, &vtx2xyz_def);
}

#[test]
fn test_hair() {
    let stiff_length = 1.0;
    let stiff_bendtwist = [1.0, 1.0, 1.0];
    let vtx2xyz_ini = del_msh_cpu::polyline3::helix(30, 0.2, 0.2, 0.5);
    let vtx2framex_ini = del_msh_cpu::polyline3::vtx2framex(&vtx2xyz_ini);
    //let (vtx2xyz_ini, vtx2framex_ini) = make_config_darboux_simple();
    let num_vtx = vtx2xyz_ini.len() / 3;
    {
        let (tril2vtxl, vtxl2xyz) =
            del_msh_cpu::polyline3::to_trimesh3_ribbon(&vtx2xyz_ini, &vtx2framex_ini, 0.1);
        del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
            "../target/hair_ribbon_ini.obj",
            &tril2vtxl,
            &vtxl2xyz,
            3,
        )
        .unwrap();
    }
    let vtx2isfix = {
        let num_vtx = vtx2xyz_ini.len() / 3;
        let mut vtx2isfix = vec![[0; 4]; num_vtx];
        vtx2isfix[0] = [1; 4];
        vtx2isfix[1] = [1, 1, 1, 0];
        /*
        for i_vtx in 0..num_vtx {
            vtx2isfix[i_vtx][3] = 1;
        }
         */
        vtx2isfix
    };
    let (mut vtx2xyz_def, mut vtx2framex_def) = {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        let mut vtx2xyz_def = vtx2xyz_ini.clone();
        let mut vtx2framex_def = vtx2framex_ini.clone();
        for i_vtx in 0..num_vtx {
            if vtx2isfix[i_vtx][0] == 0 {
                vtx2xyz_def[i_vtx * 3] += (2.0 * rng.random::<f64>() - 1.0) * 0.5;
            }
            if vtx2isfix[i_vtx][1] == 0 {
                vtx2xyz_def[i_vtx * 3 + 1] += (2.0 * rng.random::<f64>() - 1.0) * 0.5;
            }
            if vtx2isfix[i_vtx][2] == 0 {
                vtx2xyz_def[i_vtx * 3 + 2] += (2.0 * rng.random::<f64>() - 1.0) * 0.5;
            }
            if vtx2isfix[i_vtx][3] == 0 {
                vtx2framex_def[i_vtx * 3] += (2.0 * rng.random::<f64>() - 1.0) * 0.1;
                vtx2framex_def[i_vtx * 3 + 1] += (2.0 * rng.random::<f64>() - 1.0) * 0.1;
                vtx2framex_def[i_vtx * 3 + 2] += (2.0 * rng.random::<f64>() - 1.0) * 0.1;
            }
        }
        orthonormalize_framex_for_hair(&mut vtx2framex_def, &vtx2xyz_def);
        (vtx2xyz_def, vtx2framex_def)
    };
    {
        let (tril2vtxl, vtxl2xyz) =
            del_msh_cpu::polyline3::to_trimesh3_ribbon(&vtx2xyz_def, &vtx2framex_def, 0.1);
        del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
            "../target/hair_ribbon_def.obj",
            &tril2vtxl,
            &vtxl2xyz,
            3,
        )
        .unwrap();
    }
    // ----------
    // let mut col2idx = vec![usize::MAX; num_vtx];
    let mut w = 0f64;
    let mut dw = vec![[0f64; 4]; num_vtx];
    //
    let mut ddw = {
        let (vtx2idx, idx2vtx) = del_msh_cpu::polyline::vtx2vtx_rods(&[0, vtx2xyz_ini.len() / 3]);
        crate::sparse_square::Matrix::<[f64; 16]>::from_vtx2vtx(&vtx2idx, &idx2vtx)
    };
    let mut u_vec = vec![[0f64; 4]; num_vtx];
    let mut p_vec = vec![[0f64; 4]; num_vtx];
    let mut ap_vec = vec![[0f64; 4]; num_vtx];
    for _iter in 0..20 {
        let ddw_ref = crate::sparse_square::MatrixRefMut {
            num_blk: ddw.num_blk,
            row2idx: &ddw.row2idx,
            idx2col: &ddw.idx2col,
            idx2val: &mut ddw.idx2val,
            row2val: &mut ddw.row2val,
        };
        wdwddw_hair_system(
            &mut w,
            &mut dw,
            ddw_ref,
            &vtx2xyz_ini,
            &vtx2xyz_def,
            stiff_length,
            &stiff_bendtwist,
            &vtx2framex_ini,
            &vtx2framex_def,
            0.001,
        );
        // set bc flag
        for i_vtx in 0..num_vtx {
            for i_dof in 0..4 {
                if vtx2isfix[i_vtx][i_dof] == 0 {
                    continue;
                }
                dw[i_vtx][i_dof] = 0.0;
            }
        }
        ddw.set_fixed_dof::<4>(1.0, &vtx2isfix);
        dbg!(_iter, w);
        if _iter == 19 {
            assert!(w < 3.0e-6);
        }
        //
        {
            let _hist = crate::sparse_square::conjugate_gradient(
                &mut dw,
                &mut u_vec,
                &mut ap_vec,
                &mut p_vec,
                1.0e-5,
                1000,
                ddw.as_ref(),
            );
            // dbg!(hist.last().unwrap());
            update_solution_hair(&mut vtx2xyz_def, &mut vtx2framex_def, &u_vec, &vtx2isfix);
            // orthonormalize_framex_for_hair(&mut vtx2framex_def, &vtx2xyz_def);
            /*
            {
                let (tril2vtxl, vtxl2xyz) =
                    del_msh_cpu::polyline3::to_trimesh3_ribbon(&vtx2xyz_def, &vtx2framex_def, 0.1);
                del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
                    format!("../target/hair_ribbon_def_{}.obj", _iter),
                    &tril2vtxl,
                    &vtxl2xyz,
                    3,
                )
                .unwrap();
            }
             */
        }
    }
}

pub struct RodSimulator<T>
where
    T: num_traits::Float,
{
    pub vtx2xyz_ini: Vec<T>,
    pub vtx2framex_ini: Vec<T>,
    pub vtx2xyz_def: Vec<T>,
    pub vtx2framex_def: Vec<T>,
    //
    pub vtx2isfix: Vec<[i32; 4]>,
    //
    pub w: T,
    pub dw: Vec<[T; 4]>,
    pub ddw: crate::sparse_square::Matrix<[T; 16]>,
    pub conv_ratio: T,
    //
    pub stiff_length: T,
    pub stiff_bendtwist: [T; 3],
}

impl<T> RodSimulator<T>
where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
    rand::distr::StandardUniform: rand::distr::Distribution<T>,
{
    pub fn initialize_with_perturbation(&mut self, pos_mag: T, framex_mag: T) {
        use rand::SeedableRng;
        let rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        initialize_with_perturbation(
            &mut self.vtx2xyz_def,
            &mut self.vtx2framex_def,
            &self.vtx2xyz_ini,
            &self.vtx2framex_ini,
            &self.vtx2isfix,
            pos_mag,
            framex_mag,
            rng,
        );
    }

    pub fn allocate_memory_for_linear_system(&mut self) {
        let zero = T::zero();
        let num_vtx = self.vtx2xyz_ini.len() / 3;
        self.w = zero;
        self.dw = vec![[zero; 4]; num_vtx];
        //
        {
            let (vtx2idx, idx2vtx) = del_msh_cpu::polyline::vtx2vtx_rods(&[0, num_vtx]);
            self.ddw = crate::sparse_square::Matrix::<[T; 16]>::from_vtx2vtx(&vtx2idx, &idx2vtx)
        };
    }

    pub fn update_static(&mut self, pick_info: &Option<(usize, [T; 3])>) {
        let zero = T::zero();
        let one = T::one();
        let num_vtx = self.vtx2xyz_ini.len() / 3;
        crate::rod3_darboux::wdwddw_hair_system(
            &mut self.w,
            &mut self.dw,
            self.ddw.as_ref_mut(),
            &self.vtx2xyz_ini,
            &self.vtx2xyz_def,
            self.stiff_length,
            &self.stiff_bendtwist,
            &self.vtx2framex_ini,
            &self.vtx2framex_def,
            T::zero(),
        );
        if let Some((i_vtx, pos_goal)) = pick_info {
            let i_vtx = *i_vtx;
            use del_geo_core::mat4_col_major;
            let one = T::one();
            let two = one + one;
            let stiff = two * two * two * two;
            let kmat = mat4_col_major::from_diagonal(stiff, stiff, stiff, zero);
            use del_geo_core::mat4_col_major::Mat4ColMajor;
            self.ddw.row2val[i_vtx].add_in_place(&kmat);
            let c = del_geo_core::vec3::sub(
                arrayref::array_ref![self.vtx2xyz_def, i_vtx * 3, 3],
                pos_goal,
            );
            self.dw[i_vtx][0] = self.dw[i_vtx][0] + c[0];
            self.dw[i_vtx][1] = self.dw[i_vtx][1] + c[1];
            self.dw[i_vtx][2] = self.dw[i_vtx][2] + c[2];
        }
        // set bc flag
        crate::sparse_square::set_fix_dof_to_rhs_vector::<T, 4>(&mut self.dw, &self.vtx2isfix);
        self.ddw.set_fixed_dof::<4>(one, &self.vtx2isfix);
        //
        {
            let mut u_vec = vec![[zero; 4]; num_vtx];
            let mut p_vec = vec![[zero; 4]; num_vtx];
            let mut ap_vec = vec![[zero; 4]; num_vtx];
            let _hist = crate::sparse_square::conjugate_gradient(
                &mut self.dw,
                &mut u_vec,
                &mut ap_vec,
                &mut p_vec,
                self.conv_ratio,
                1000,
                self.ddw.as_ref(),
            );
            // dbg!(hist.last().unwrap());
            update_solution_hair(
                &mut self.vtx2xyz_def,
                &mut self.vtx2framex_def,
                &u_vec,
                &self.vtx2isfix,
            );
        }
    }
}
