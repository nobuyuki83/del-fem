/// frame after small vertex movement of z0 and z-axis rotation
fn updated_rod_frame<T>(ux0: &[T; 3], z0: &[T; 3], dz: &[T; 3], dtheta: T) -> [[T; 3]; 3]
where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
    use del_geo_core::vec3::Vec3;
    let lenz = z0.norm();
    let lenzinv = T::one() / lenz;
    let uz0 = z0.scale(lenzinv);
    let uy0 = uz0.cross(ux0);
    // ux1 = exp{ skew(z) } * ux0 = { I + skew(z) } * ux0
    let ux1 = ux0.scale(dtheta.cos()).add(&uy0.scale(dtheta.sin()));
    // uy1 = exp{ skew(z) } * uy0 = { I + skew(z) } * uy0
    let uy1 = uy0.scale(dtheta.cos()).sub(&ux0.scale(dtheta.sin()));
    let r_mat = del_geo_core::mat3_col_major::from_axisangle_vec(&uz0.cross(dz).scale(lenzinv));
    let ux2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &ux1);
    let uy2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &uy1);
    let uz2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &uz0);
    [ux2, uy2, uz2]
}

/// # Argument
/// * `frame` - `frame[i][j]` is `frame[i]` is the `i`-th axis and `j`-th coordinate
/// # Return
/// * `dfdv` - differentiation of frame w.r.t vertex position (i.e., `frame[2] * len`) )
///   * `dfdv[i][j][k]` - differentiation of `frame[i][j]` w.r.t vertex position `v[k]`
fn rod_frame_gradient(length: f64, frame: &[[f64; 3]; 3]) -> ([[[f64; 3]; 3]; 3], [[f64; 3]; 3]) {
    use del_geo_core::mat3_col_major;
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    let leninv = 1.0 / length;
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
    let dfdt = [frame[1], frame[0].scale(-1.0), [0f64; 3]];
    (dfdv, dfdt)
}

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
        let w2 = [q.dot(&frm2[0]), q.dot(&frm2[1]), q.dot(&frm2[2])];
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
        for i_axis in 0..3 {
            let (ddw2ddv, ddwdtdv, ddwddt) = rod_frame_hessian(i_axis, len2, &q, &frm2);
            let a = (w0[i_axis] + w4[i_axis] - 2.0 * w2[i_axis]) / (eps * eps);
            let b = del_geo_core::mat3_col_major::mult_vec(&ddw2ddv, &du).dot(&du)
                + ddwddt * dt * dt
                + 2.0 * ddwdtdv.dot(&du) * dt;
            let err = (a - b).abs() / (a.abs() + b.abs() + 3.0e-3);
            assert!(err < 3.0e-3, "{} {} {}  {}", a, b, (a - b).abs(), err);
        }
    }
}

// above gradient and hessian of frame
// ---------------------------------------

// add derivative of dot( Frm0[i], Frm1[j] ) with respect to the 3 points and 2 rotations
// of the rod element
pub fn add_gradient_of_dot_frame_axis(
    dvdp: &mut [[f64; 3]; 3],
    dvdt: &mut [f64; 2],
    c: f64,
    i0_axis: usize,
    frma: &[[f64; 3]; 3],
    dfadp: &[[[f64; 3]; 3]; 3],
    dfadt: &[[f64; 3]; 3],
    i1_axis: usize,
    frmb: &[[f64; 3]; 3],
    dfbdp: &[[[f64; 3]; 3]; 3],
    dfbdt: &[[f64; 3]; 3],
) {
    use del_geo_core::vec3::Vec3;
    dvdt[0] += c * frmb[i1_axis].dot(&dfadt[i0_axis]);
    dvdt[1] += c * frma[i0_axis].dot(&dfbdt[i1_axis]);
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

/// Darboux vector in the reference configuration and its gradient
pub fn cdc_rod_darboux(
    p: &[[f64; 3]; 3],
    x: &[[f64; 3]; 2],
) -> ([f64; 3], [[[f64; 3]; 3]; 3], [[f64; 2]; 3]) {
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
    let s = 1.0 + frma[0].dot(&frmb[0]) + frma[1].dot(&frmb[1]) + frma[2].dot(&frmb[2]);
    let (dsdp, dsdt) = {
        // making derivative of Y
        let mut dsdp = [[0f64; 3]; 3];
        let mut dsdt = [0f64; 2];
        add_gradient_of_dot_frame_axis(
            &mut dsdp, &mut dsdt, 1., 0, &frma, &dfadp, &dfadt, 0, &frmb, &dfbdp, &dfbdt,
        );
        add_gradient_of_dot_frame_axis(
            &mut dsdp, &mut dsdt, 1., 1, &frma, &dfadp, &dfadt, 1, &frmb, &dfbdp, &dfbdt,
        );
        add_gradient_of_dot_frame_axis(
            &mut dsdp, &mut dsdt, 1., 2, &frma, &dfadp, &dfadt, 2, &frmb, &dfbdp, &dfbdt,
        );
        (dsdp, dsdt)
    };
    let mut c = [0f64; 3];
    let mut dcdp = [[[0f64; 3]; 3]; 3];
    let mut dcdt = [[0f64; 2]; 3];
    for iaxis in 0..3 {
        let jaxis = (iaxis + 1) % 3;
        let kaxis = (iaxis + 2) % 3;
        let u = frma[jaxis].dot(&frmb[kaxis]) - frma[kaxis].dot(&frmb[jaxis]);
        let mut dudp = [[0f64; 3]; 3];
        let mut dudt = [0.0, 0.0];
        {
            add_gradient_of_dot_frame_axis(
                &mut dudp, &mut dudt, 1., jaxis, &frma, &dfadp, &dfadt, kaxis, &frmb, &dfbdp,
                &dfbdt,
            );
            add_gradient_of_dot_frame_axis(
                &mut dudp, &mut dudt, -1., kaxis, &frma, &dfadp, &dfadt, jaxis, &frmb, &dfbdp,
                &dfbdt,
            );
        }
        c[iaxis] = u / s;
        {
            let t0 = 1.0 / s;
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
            assert!(err < 4.0e-6, "{}", err);
        }
    }
}

fn wdwdwdw_darboux_rod_hair_approx_hessian(
    p: &[[f64; 3]; 3],
    x: &[[f64; 3]; 2],
    stiff_bendtwist: &[f64; 3],
    darboux0: &[f64; 3],
) -> (f64, [[f64; 4]; 3], [[[f64; 16]; 3]; 3]) {
    use del_geo_core::vec3::Vec3;
    let (c, dcdp, dcdt) = cdc_rod_darboux(p, x);
    let r = [c[0] - darboux0[0], c[1] - darboux0[1], c[2] - darboux0[2]];
    let w = 0.5
        * (stiff_bendtwist[0] * r[0] * r[0]
            + stiff_bendtwist[1] * r[1] * r[1]
            + stiff_bendtwist[2] * r[2] * r[2]);
    let dw = {
        let mut dw = [[0f64; 4]; 3];
        for ino in 0..3 {
            let t0 = dcdp[0][ino].scale(r[0] * stiff_bendtwist[0]);
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
        dw[2][3] = 0.0;
        dw
    };
    let ddw = {
        let mut ddw = [[[0f64; 16]; 3]; 3];
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
                ddw[ino][jno] = del_geo_core::mat4_col_major::from_mat3_col_major_adding_w(&m);
            }
        }
        for ino in 0..3 {
            for jno in 0..2 {
                let v0 = dcdp[0][ino].scale(stiff_bendtwist[0] * dcdt[0][jno]);
                let v1 = dcdp[1][ino].scale(stiff_bendtwist[1] * dcdt[1][jno]);
                let v2 = dcdp[2][ino].scale(stiff_bendtwist[2] * dcdt[2][jno]);
                let v = del_geo_core::vec3::add_three(&v0, &v1, &v2);
                ddw[ino][jno][3] = v[0];
                ddw[jno][ino][12] = v[0];
                ddw[jno][ino][13] = v[1];
                ddw[ino][jno][7] = v[1];
                ddw[ino][jno][11] = v[2];
                ddw[jno][ino][14] = v[2]
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
        let stiff_bendtwist = [1.0, 1.0, 1.0];
        let darboux_ini = [0.0, 0.0, 0.0];
        let (w2, dw2dpt, ddw2ddpt) =
            wdwdwdw_darboux_rod_hair_approx_hessian(&p2, &x2, &stiff_bendtwist, &darboux_ini);
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
        dbg!(dw_ana, dw_num, err);
        assert!(err < 4.0e-5, "{}", err);
    }
}
