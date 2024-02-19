use num_traits::AsPrimitive;

/// plate bending energy
///
/// * `p` - initial XY position
/// * `u` - z displacement and xy axis rotation
pub fn w_dw_ddw_plate_bending<T>(
    p: &[[T; 2]; 3],
    u: &[[T; 3]; 3],
    thk: T,
    lambda: T,
    myu: T)
    where T: num_traits::Float + 'static + Copy + std::ops::MulAssign + std::ops::AddAssign,
          f64: AsPrimitive<T>
{
    let zero = T::zero();
    let two = T::one() + T::one();
    let four = two + two;
    let half = T::one() / two;
    let onefourth = T::one() / four;
    //
    let area = del_geo::tri2::area_(&p[0], &p[1], &p[2]);
    let Gd: [[T; 3]; 3] = [ // undeformed edge vector
        [p[1][0] - p[0][0], p[1][1] - p[0][1], zero],
        [p[2][0] - p[0][0], p[2][1] - p[0][1], zero],
        [zero, zero, 0.564.as_() * thk]
    ];

    let Gu: [[T; 3]; 3] = { // inverse of Gd
        let mut Gu = [[zero; 3]; 3];
        del_geo::vec3::cross_mut_(&mut Gu[0], &Gd[1], &Gd[2]);
        let invtmp1 = T::one() / del_geo::vec3::dot_(&Gu[0], &Gd[0]);
        Gu[0][0] *= invtmp1;
        Gu[0][1] *= invtmp1;
        Gu[0][2] *= invtmp1;
        //
        del_geo::vec3::cross_mut_(&mut Gu[1], &Gd[2], &Gd[0]);
        let invtmp2 = T::one() / del_geo::vec3::dot_(&Gu[1], &Gd[1]);
        Gu[1][0] *= invtmp2;
        Gu[1][1] *= invtmp2;
        Gu[1][2] *= invtmp2;
        //
        del_geo::vec3::cross_mut_(&mut Gu[2], &Gd[0], &Gd[1]);
        let invtmp3 = T::one() / del_geo::vec3::dot_(&Gu[2], &Gd[2]);
        Gu[2][0] *= invtmp3;
        Gu[2][1] *= invtmp3;
        Gu[2][2] *= invtmp3;
        Gu
    };

    let GuGu2: [T; 4] = [
        del_geo::vec3::dot_(&Gu[0], &Gu[0]), // rr 0
        del_geo::vec3::dot_(&Gu[1], &Gu[1]), // ss 1
        del_geo::vec3::dot_(&Gu[0], &Gu[1]), // sr 2
        del_geo::vec3::dot_(&Gu[2], &Gu[2]), // tt 3
    ];

    let mut w = zero;
    let mut dw = [[zero; 3]; 3];
    let mut ddw = [[[zero; 9]; 3]; 3];
    {
        let CnstA: [[T; 3]; 3] = [ // {rr,ss,sr} x {rr,ss,sr}
            [
                lambda * GuGu2[0] * GuGu2[0] + two * myu * (GuGu2[0] * GuGu2[0]), // 00(0):00(0) 00(0):00(0)
                lambda * GuGu2[0] * GuGu2[1] + two * myu * (GuGu2[2] * GuGu2[2]), // 00(0):11(1) 01(2):01(2)
                lambda * GuGu2[0] * GuGu2[2] + two * myu * (GuGu2[0] * GuGu2[2])
            ],// 00(0):01(2) 00(0):01(2)
            [
                lambda * GuGu2[1] * GuGu2[0] + two * myu * (GuGu2[2] * GuGu2[2]), // 11(1):00(0) 01(2):01(2)
                lambda * GuGu2[1] * GuGu2[1] + two * myu * (GuGu2[1] * GuGu2[1]), // 11(1):11(1) 11(1):11(1)
                lambda * GuGu2[1] * GuGu2[2] + two * myu * (GuGu2[1] * GuGu2[2])
            ],// 11(1):01(2) 11(1):01(2)
            [
                lambda * GuGu2[2] * GuGu2[0] + two * myu * (GuGu2[0] * GuGu2[2]), // 01(2):00(0) 00(0):01(2)
                lambda * GuGu2[2] * GuGu2[1] + two * myu * (GuGu2[2] * GuGu2[1]), // 01(2):11(1) 11(1):01(2)
                lambda * GuGu2[2] * GuGu2[2] + myu * (GuGu2[0] * GuGu2[1] + GuGu2[2] * GuGu2[2])
            ] // 01(2):01(2) 00(0):11(1) 01(2):01(2)
        ];
        let CnstB: [[T; 3]; 3] = [ // {rr,ss,sr} x {rr,ss,sr}
            [CnstA[0][0], CnstA[0][1], two * CnstA[0][2]],
            [CnstA[1][0], CnstA[1][1], two * CnstA[1][2]],
            [two * CnstA[2][0], two * CnstA[2][1], four * CnstA[2][2]],
        ];
        let EA0t = (Gd[0][0] * (u[1][2] - u[0][2]) - Gd[0][1] * (u[1][1] - u[0][1])) * half * thk;
        let EA1t = (Gd[1][0] * (u[2][2] - u[0][2]) - Gd[1][1] * (u[2][1] - u[0][1])) * half * thk;
        let EA2t = (Gd[0][0] * (u[2][2] - u[0][2]) - Gd[0][1] * (u[2][1] - u[0][1])
            + Gd[1][0] * (u[1][2] - u[0][2]) - Gd[1][1] * (u[1][1] - u[0][1])) * onefourth * thk;
        ////
        for i_integration in 0..2 {
            let t0 = if i_integration == 0 { -T::one() / 3f64.sqrt().as_() } else { T::one() / 3f64.sqrt().as_() };
            let w_integration = area * thk / two;
            let E: [T; 3] = [t0 * EA0t, t0 * EA1t, t0 * EA2t];
            let dE: [[[T; 3]; 3]; 3] = [
                [
                    [zero, Gd[0][1] * half * thk * t0, -Gd[0][0] * half * thk * t0],
                    [zero, -Gd[0][1] * half * thk * t0, Gd[0][0] * half * thk * t0],
                    [zero, zero, zero]],
                [
                    [zero, Gd[1][1] * half * thk * t0, -Gd[1][0] * half * thk * t0],
                    [zero, zero, zero],
                    [zero, -Gd[1][1] * half * thk * t0, Gd[1][0] * half * thk * t0]
                ],
                [
                    [zero, (Gd[0][1] + Gd[1][1]) * onefourth * thk * t0, -(Gd[0][0] + Gd[1][0]) * onefourth * thk * t0],
                    [zero, -Gd[1][1] * onefourth * thk * t0, Gd[1][0] * onefourth * thk * t0],
                    [zero, -Gd[0][1] * onefourth * thk * t0, Gd[0][0] * onefourth * thk * t0]
                ]
            ];
            ////
            let SB: [T; 3] = [
                CnstB[0][0] * E[0] + CnstB[0][1] * E[1] + CnstB[0][2] * E[2],
                CnstB[1][0] * E[0] + CnstB[1][1] * E[1] + CnstB[1][2] * E[2],
                CnstB[2][0] * E[0] + CnstB[2][1] * E[1] + CnstB[2][2] * E[2]
            ];
            w += w_integration * half * (E[0] * SB[0] + E[1] * SB[1] + E[2] * SB[2]);
            for (ino, idof) in itertools::iproduct!(0..3, 0..3) {
                dw[ino][idof] += w_integration * (
                    SB[0] * dE[0][ino][idof]
                        + SB[1] * dE[1][ino][idof]
                        + SB[2] * dE[2][ino][idof]);
            }
            for (ino, jno, idof, jdof) in itertools::iproduct!(
                0..3, 0..3, 0..3, 0..3) {
                let dtmp =
                    dE[0][ino][idof] * CnstB[0][0] * dE[0][jno][jdof]
                        + dE[0][ino][idof] * CnstB[0][1] * dE[1][jno][jdof]
                        + dE[0][ino][idof] * CnstB[0][2] * dE[2][jno][jdof]
                        + dE[1][ino][idof] * CnstB[1][0] * dE[0][jno][jdof]
                        + dE[1][ino][idof] * CnstB[1][1] * dE[1][jno][jdof]
                        + dE[1][ino][idof] * CnstB[1][2] * dE[2][jno][jdof]
                        + dE[2][ino][idof] * CnstB[2][0] * dE[0][jno][jdof]
                        + dE[2][ino][idof] * CnstB[2][1] * dE[1][jno][jdof]
                        + dE[2][ino][idof] * CnstB[2][2] * dE[2][jno][jdof];
                ddw[ino][jno][idof * 3 + jdof] += w_integration * dtmp;
            }
        }
    }
    {
        let CnstA: [[T; 2]; 2] = [ // {rt,st} x {rt,st}
            [
                myu * GuGu2[0] * GuGu2[3],  // rt*rt -> rr(0):tt(3)
                myu * GuGu2[2] * GuGu2[3]
            ], // st*rt -> sr(2):tt(3)
            [
                myu * GuGu2[2] * GuGu2[3],  // rt*st -> rs(2):tt(3)
                myu * GuGu2[1] * GuGu2[3]
            ]  // st*st -> ss(1):tt(3)
        ];
        let CnstB: [[T; 2]; 2] = [
            [four * CnstA[0][0], two * CnstA[0][1]],
            [two * CnstA[1][0], four * CnstA[1][1]]
        ];
        let Ert_01 = half * thk * (u[1][0] - u[0][0] + half * Gd[0][0] * (u[0][2] + u[1][2]) - half * Gd[0][1] * (u[0][1] + u[1][1]));
        let Ert_12 = half * thk * (u[1][0] - u[0][0] + half * Gd[0][0] * (u[1][2] + u[2][2]) - half * Gd[0][1] * (u[1][1] + u[2][1]));
        let Est_12 = half * thk * (u[2][0] - u[0][0] + half * Gd[1][0] * (u[1][2] + u[2][2]) - half * Gd[1][1] * (u[1][1] + u[2][1]));
        let Est_20 = half * thk * (u[2][0] - u[0][0] + half * Gd[1][0] * (u[2][2] + u[0][2]) - half * Gd[1][1] * (u[2][1] + u[0][1]));
        let dErt_01: [[T; 3]; 3] = [
            [-half * thk, -onefourth * thk * Gd[0][1], onefourth * thk * Gd[0][0]],
            [half * thk, -onefourth * thk * Gd[0][1], onefourth * thk * Gd[0][0]],
            [zero, zero, zero]];
        let dEst_20: [[T; 3]; 3] = [
            [-half * thk, -onefourth * thk * Gd[1][1], onefourth * thk * Gd[1][0]],
            [zero, zero, zero],
            [half * thk, -onefourth * thk * Gd[1][1], onefourth * thk * Gd[1][0]]];
        let dEC: [[T; 3]; 3] = [
            [zero, onefourth * thk * Gd[0][1] - onefourth * thk * Gd[1][1], -onefourth * thk * Gd[0][0] + onefourth * thk * Gd[1][0]],
            [zero, onefourth * thk * Gd[1][1], -onefourth * thk * Gd[1][0]],
            [zero, -onefourth * thk * Gd[0][1], onefourth * thk * Gd[0][0]]];
        let CE = (Ert_12 - Ert_01) - (Est_12 - Est_20);
        let pos_integration: [[T; 2]; 3] = [[half, zero], [half, half], [zero, half]]; // position to integrate
        for i_integration in 0..3 {
            let r = pos_integration[i_integration][0];
            let s = pos_integration[i_integration][1];
            let w_integration = area * thk / (T::one() + two);
            let E: [T; 2] = [Ert_01 + CE * s, Est_20 - CE * r];
            let mut dE = [[[T::zero(); 3]; 3]; 2];
            for (ino, idof) in itertools::iproduct!(0..3, 0..3) {
                dE[0][ino][idof] = dErt_01[ino][idof] + dEC[ino][idof] * s;
                dE[1][ino][idof] = dEst_20[ino][idof] - dEC[ino][idof] * r;
            }
            let SB: [T; 2] = [
                CnstB[0][0] * E[0] + CnstB[0][1] * E[1],
                CnstB[1][0] * E[0] + CnstB[1][1] * E[1]
            ];
            w += w_integration * half * (SB[0] * E[0] + SB[1] * E[1]);
            for (ino, idof) in itertools::iproduct!(0..3, 0..3) {
                dw[ino][idof] += w_integration * (SB[0] * dE[0][ino][idof] + SB[1] * dE[1][ino][idof]);
            }
            for (ino, jno, idof, jdof) in itertools::iproduct!(0..3, 0..3, 0..3, 0..3) {
                let dtmp = dE[0][ino][idof] * CnstB[0][0] * dE[0][jno][jdof]
                    + dE[0][ino][idof] * CnstB[0][1] * dE[1][jno][jdof]
                    + dE[1][ino][idof] * CnstB[1][0] * dE[0][jno][jdof]
                    + dE[1][ino][idof] * CnstB[1][1] * dE[1][jno][jdof];
                ddw[ino][jno][idof * 3 + jdof] += w_integration * dtmp;
            }
        }
    }
}
