use num_traits::AsPrimitive;

/// plate bending energy
///
/// * `p` - initial XY position
/// * `u` - z displacement and xy-axis rotation
pub fn w_dw_ddw_plate_bending<T>(
    p: &[[T; 2]; 3],
    u: &[[T; 3]; 3],
    thk: T,
    lambda: T,
    myu: T,
) -> (T, [[T; 3]; 3], [[[T; 9]; 3]; 3])
where
    T: num_traits::Float + std::ops::MulAssign + std::ops::AddAssign + 'static + Copy,
    f64: AsPrimitive<T>,
{
    let zero = T::zero();
    let two = T::one() + T::one();
    let four = two + two;
    let half = T::one() / two;
    let onefourth = T::one() / four;
    //
    let area = del_geo::tri2::area_(&p[0], &p[1], &p[2]);
    let gd: [[T; 3]; 3] = [
        // covariant basis vectors
        [p[1][0] - p[0][0], p[1][1] - p[0][1], zero],
        [p[2][0] - p[0][0], p[2][1] - p[0][1], zero],
        [zero, zero, 0.564.as_() * thk],
    ];

    let gu: [[T; 3]; 3] = del_geo::mat3::inverse_array_of_array(&gd);

    let gu_gu: [T; 4] = [
        del_geo::vec3::dot_(&gu[0], &gu[0]), // rr 0
        del_geo::vec3::dot_(&gu[1], &gu[1]), // ss 1
        del_geo::vec3::dot_(&gu[0], &gu[1]), // sr 2
        del_geo::vec3::dot_(&gu[2], &gu[2]), // tt 3
    ];

    let mut w = zero;
    let mut dw = [[zero; 3]; 3];
    let mut ddw = [[[zero; 9]; 3]; 3];
    {
        // integrate transverse sheer stress
        let coeffa: [[T; 3]; 3] = [
            // {rr,ss,sr} x {rr,ss,sr}
            [
                lambda * gu_gu[0] * gu_gu[0] + two * myu * (gu_gu[0] * gu_gu[0]), // 00(0):00(0) 00(0):00(0)
                lambda * gu_gu[0] * gu_gu[1] + two * myu * (gu_gu[2] * gu_gu[2]), // 00(0):11(1) 01(2):01(2)
                lambda * gu_gu[0] * gu_gu[2] + two * myu * (gu_gu[0] * gu_gu[2]),
            ], // 00(0):01(2) 00(0):01(2)
            [
                lambda * gu_gu[1] * gu_gu[0] + two * myu * (gu_gu[2] * gu_gu[2]), // 11(1):00(0) 01(2):01(2)
                lambda * gu_gu[1] * gu_gu[1] + two * myu * (gu_gu[1] * gu_gu[1]), // 11(1):11(1) 11(1):11(1)
                lambda * gu_gu[1] * gu_gu[2] + two * myu * (gu_gu[1] * gu_gu[2]),
            ], // 11(1):01(2) 11(1):01(2)
            [
                lambda * gu_gu[2] * gu_gu[0] + two * myu * (gu_gu[0] * gu_gu[2]), // 01(2):00(0) 00(0):01(2)
                lambda * gu_gu[2] * gu_gu[1] + two * myu * (gu_gu[2] * gu_gu[1]), // 01(2):11(1) 11(1):01(2)
                lambda * gu_gu[2] * gu_gu[2] + myu * (gu_gu[0] * gu_gu[1] + gu_gu[2] * gu_gu[2]),
            ], // 01(2):01(2) 00(0):11(1) 01(2):01(2)
        ];
        let coeffb: [[T; 3]; 3] = [
            // {rr,ss,sr} x {rr,ss,sr}
            [coeffa[0][0], coeffa[0][1], two * coeffa[0][2]],
            [coeffa[1][0], coeffa[1][1], two * coeffa[1][2]],
            [two * coeffa[2][0], two * coeffa[2][1], four * coeffa[2][2]],
        ];
        // covariant coefficients of linear strain
        let de_rr_dt =
            (gd[0][0] * (u[1][2] - u[0][2]) - gd[0][1] * (u[1][1] - u[0][1])) * half * thk;
        let de_ss_dt =
            (gd[1][0] * (u[2][2] - u[0][2]) - gd[1][1] * (u[2][1] - u[0][1])) * half * thk;
        let de_rs_dt = (gd[0][0] * (u[2][2] - u[0][2]) - gd[0][1] * (u[2][1] - u[0][1])
            + gd[1][0] * (u[1][2] - u[0][2])
            - gd[1][1] * (u[1][1] - u[0][1]))
            * onefourth
            * thk;
        ////
        for i_integration in 0..2 {
            // integration of energy related to transverse shear strain
            let t0 = if i_integration == 0 {
                -T::one() / 3f64.sqrt().as_()
            } else {
                T::one() / 3f64.sqrt().as_()
            };
            let w_integration = area * thk / two;
            let strain: [T; 3] = [t0 * de_rr_dt, t0 * de_ss_dt, t0 * de_rs_dt]; // linear strain (e_rr, e_ss, e_rs)
            let dstrain: [[[T; 3]; 3]; 3] = [
                [
                    [
                        zero,
                        gd[0][1] * half * thk * t0,
                        -gd[0][0] * half * thk * t0,
                    ],
                    [
                        zero,
                        -gd[0][1] * half * thk * t0,
                        gd[0][0] * half * thk * t0,
                    ],
                    [zero, zero, zero],
                ],
                [
                    [
                        zero,
                        gd[1][1] * half * thk * t0,
                        -gd[1][0] * half * thk * t0,
                    ],
                    [zero, zero, zero],
                    [
                        zero,
                        -gd[1][1] * half * thk * t0,
                        gd[1][0] * half * thk * t0,
                    ],
                ],
                [
                    [
                        zero,
                        (gd[0][1] + gd[1][1]) * onefourth * thk * t0,
                        -(gd[0][0] + gd[1][0]) * onefourth * thk * t0,
                    ],
                    [
                        zero,
                        -gd[1][1] * onefourth * thk * t0,
                        gd[1][0] * onefourth * thk * t0,
                    ],
                    [
                        zero,
                        -gd[0][1] * onefourth * thk * t0,
                        gd[0][0] * onefourth * thk * t0,
                    ],
                ],
            ];
            ////
            let stress: [T; 3] = [
                coeffb[0][0] * strain[0] + coeffb[0][1] * strain[1] + coeffb[0][2] * strain[2],
                coeffb[1][0] * strain[0] + coeffb[1][1] * strain[1] + coeffb[1][2] * strain[2],
                coeffb[2][0] * strain[0] + coeffb[2][1] * strain[1] + coeffb[2][2] * strain[2],
            ];
            w += w_integration
                * half
                * (strain[0] * stress[0] + strain[1] * stress[1] + strain[2] * stress[2]);
            for (ino, idof) in itertools::iproduct!(0..3, 0..3) {
                dw[ino][idof] += w_integration
                    * (stress[0] * dstrain[0][ino][idof]
                        + stress[1] * dstrain[1][ino][idof]
                        + stress[2] * dstrain[2][ino][idof]);
            }
            for (ino, jno, idof, jdof) in itertools::iproduct!(0..3, 0..3, 0..3, 0..3) {
                let dtmp = dstrain[0][ino][idof] * coeffb[0][0] * dstrain[0][jno][jdof]
                    + dstrain[0][ino][idof] * coeffb[0][1] * dstrain[1][jno][jdof]
                    + dstrain[0][ino][idof] * coeffb[0][2] * dstrain[2][jno][jdof]
                    + dstrain[1][ino][idof] * coeffb[1][0] * dstrain[0][jno][jdof]
                    + dstrain[1][ino][idof] * coeffb[1][1] * dstrain[1][jno][jdof]
                    + dstrain[1][ino][idof] * coeffb[1][2] * dstrain[2][jno][jdof]
                    + dstrain[2][ino][idof] * coeffb[2][0] * dstrain[0][jno][jdof]
                    + dstrain[2][ino][idof] * coeffb[2][1] * dstrain[1][jno][jdof]
                    + dstrain[2][ino][idof] * coeffb[2][2] * dstrain[2][jno][jdof];
                ddw[ino][jno][idof * 3 + jdof] += w_integration * dtmp;
            }
        }
    }
    {
        let coeffa: [[T; 2]; 2] = [
            // {rt,st} x {rt,st}
            [
                myu * gu_gu[0] * gu_gu[3], // rt*rt -> rr(0):tt(3)
                myu * gu_gu[2] * gu_gu[3],
            ], // st*rt -> sr(2):tt(3)
            [
                myu * gu_gu[2] * gu_gu[3], // rt*st -> rs(2):tt(3)
                myu * gu_gu[1] * gu_gu[3],
            ], // st*st -> ss(1):tt(3)
        ];
        let coeffb: [[T; 2]; 2] = [
            [four * coeffa[0][0], two * coeffa[0][1]],
            [two * coeffa[1][0], four * coeffa[1][1]],
        ];
        let e_rt_01 = half
            * thk
            * (u[1][0] - u[0][0] + half * gd[0][0] * (u[0][2] + u[1][2])
                - half * gd[0][1] * (u[0][1] + u[1][1]));
        let e_rt_12 = half
            * thk
            * (u[1][0] - u[0][0] + half * gd[0][0] * (u[1][2] + u[2][2])
                - half * gd[0][1] * (u[1][1] + u[2][1]));
        let e_st_12 = half
            * thk
            * (u[2][0] - u[0][0] + half * gd[1][0] * (u[1][2] + u[2][2])
                - half * gd[1][1] * (u[1][1] + u[2][1]));
        let e_st_20 = half
            * thk
            * (u[2][0] - u[0][0] + half * gd[1][0] * (u[2][2] + u[0][2])
                - half * gd[1][1] * (u[2][1] + u[0][1]));
        let de_rt_01: [[T; 3]; 3] = [
            [
                -half * thk,
                -onefourth * thk * gd[0][1],
                onefourth * thk * gd[0][0],
            ],
            [
                half * thk,
                -onefourth * thk * gd[0][1],
                onefourth * thk * gd[0][0],
            ],
            [zero, zero, zero],
        ];
        let de_st_20: [[T; 3]; 3] = [
            [
                -half * thk,
                -onefourth * thk * gd[1][1],
                onefourth * thk * gd[1][0],
            ],
            [zero, zero, zero],
            [
                half * thk,
                -onefourth * thk * gd[1][1],
                onefourth * thk * gd[1][0],
            ],
        ];
        let coeff_e_rt = (e_rt_12 - e_rt_01) - (e_st_12 - e_st_20);
        let dcoeff_e_rt: [[T; 3]; 3] = [
            [
                zero,
                onefourth * thk * gd[0][1] - onefourth * thk * gd[1][1],
                -onefourth * thk * gd[0][0] + onefourth * thk * gd[1][0],
            ],
            [
                zero,
                onefourth * thk * gd[1][1],
                -onefourth * thk * gd[1][0],
            ],
            [
                zero,
                -onefourth * thk * gd[0][1],
                onefourth * thk * gd[0][0],
            ],
        ];
        let pos_integration = [(half, zero), (half, half), (zero, half)]; // position to integrate
        for (r, s) in pos_integration {
            let w_integration = area * thk / (T::one() + two);
            let strain: [T; 2] = [e_rt_01 + coeff_e_rt * s, e_st_20 - coeff_e_rt * r];
            let mut dstrain = [[[T::zero(); 3]; 3]; 2];
            for (ino, idof) in itertools::iproduct!(0..3, 0..3) {
                dstrain[0][ino][idof] = de_rt_01[ino][idof] + dcoeff_e_rt[ino][idof] * s;
                dstrain[1][ino][idof] = de_st_20[ino][idof] - dcoeff_e_rt[ino][idof] * r;
            }
            let stress: [T; 2] = [
                coeffb[0][0] * strain[0] + coeffb[0][1] * strain[1],
                coeffb[1][0] * strain[0] + coeffb[1][1] * strain[1],
            ];
            w += w_integration * half * (stress[0] * strain[0] + stress[1] * strain[1]);
            for (ino, idof) in itertools::iproduct!(0..3, 0..3) {
                dw[ino][idof] += w_integration
                    * (stress[0] * dstrain[0][ino][idof] + stress[1] * dstrain[1][ino][idof]);
            }
            for (ino, jno, idof, jdof) in itertools::iproduct!(0..3, 0..3, 0..3, 0..3) {
                let dtmp = dstrain[0][ino][idof] * coeffb[0][0] * dstrain[0][jno][jdof]
                    + dstrain[0][ino][idof] * coeffb[0][1] * dstrain[1][jno][jdof]
                    + dstrain[1][ino][idof] * coeffb[1][0] * dstrain[0][jno][jdof]
                    + dstrain[1][ino][idof] * coeffb[1][1] * dstrain[1][jno][jdof];
                ddw[ino][jno][idof * 3 + jdof] += w_integration * dtmp;
            }
        }
    }
    (w, dw, ddw)
}

#[test]
fn test_w_dw_ddw_plate_bending() {
    type T = f64;
    let p: [[T; 2]; 3] = [[0., 0.], [1., 0.1], [0.1, 1.0]];
    let u0: [[T; 3]; 3] = [[0.01, 0.02, 0.03], [-0.01, 0.01, 0.03], [0.02, 0.03, 0.02]];
    let thickness1 = 0.03;
    let lambda1 = 1.0;
    let myu1 = 1.0;
    let eps = 1.0e-5;
    // -------------------
    let (w0, dw0, ddw0) = w_dw_ddw_plate_bending(&p, &u0, thickness1, lambda1, myu1);
    for (ino, idof) in itertools::iproduct!(0..3, 0..3) {
        let u1 = {
            let mut u1 = u0.clone();
            u1[ino][idof] += eps;
            u1
        };
        let (w1, dw1, _ddw1) = w_dw_ddw_plate_bending(&p, &u1, thickness1, lambda1, myu1);
        {
            let v0 = (w1 - w0) / eps;
            let v1 = dw0[ino][idof];
            assert!((v0 - v1).abs() < (1. + v1.abs()) * 1.0e-4);
        }
        for (jno, jdof) in itertools::iproduct!(0..3, 0..3) {
            let v0 = (dw1[jno][jdof] - dw0[jno][jdof]) / eps;
            let v1 = ddw0[ino][jno][idof * 3 + jdof];
            assert!((v0 - v1).abs() < (1. + v1.abs()) * 1.0e-4);
        }
    }
}

pub fn mass_lumped_plate_bending<T>(
    tri2vtx: &[usize],
    vtx2xy: &[T],
    thick: T,
    rho: T,
    vtx2mass: &mut [T],
) where
    T: num_traits::Float + std::ops::AddAssign,
{
    assert_eq!(vtx2mass.len(), vtx2xy.len() / 2 * 3);
    vtx2mass.fill(T::zero());
    let three = T::one() + T::one() + T::one();
    let four = three + T::one();
    let twelve = three * four;
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = vtx2xy[i0 * 2..i0 * 2 + 2].try_into().unwrap();
        let p1 = vtx2xy[i1 * 2..i1 * 2 + 2].try_into().unwrap();
        let p2 = vtx2xy[i2 * 2..i2 * 2 + 2].try_into().unwrap();
        let a012 = del_geo::tri2::area_(p0, p1, p2);
        let m0 = a012 / three * rho * thick;
        let m1 = a012 / three * rho * thick * thick * thick / twelve;
        for i_vtx in node2vtx {
            vtx2mass[i_vtx * 3 + 0] += m0;
            vtx2mass[i_vtx * 3 + 1] += m1;
            vtx2mass[i_vtx * 3 + 2] += m1;
        }
    }
}
