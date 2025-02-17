

pub fn wdwddw_energy_density_from_scale_arap<Real>(s: &[Real; 3]) -> (Real, [Real; 3], [Real; 9])
where
    Real: num_traits::Float,
{
    let one = Real::one();
    let two = one + one;
    let eng = (s[0] - one).powi(2) + (s[1] - one).powi(2) + (s[2] - one).powi(2);
    let deng = [two * (s[0] - one), two * (s[1] - one), two * (s[2] - one)];
    (
        eng,
        deng,
        del_geo_core::mat3_col_major::from_diagonal(&[two, two, two]),
    )
}

#[test]
fn test_energy_density_from_scale_arap() -> anyhow::Result<()> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    for _iter in 0..100 {
        let f0: [f64; 9] = std::array::from_fn(|_| rng.random::<f64>());
        let (u0, s0, v0) = del_geo_core::mat3_col_major::svd(
            &f0,
            del_geo_core::mat3_sym::EigenDecompositionModes::JacobiNumIter(100),
        )
            .unwrap();
        let (w0, dw0ds, ddw0ds) = wdwddw_energy_density_from_scale_arap(&s0);
        let (ds0, dds0) = del_geo_core::mat3_col_major::gradient_and_hessian_of_svd_scale(&u0, &s0, &v0);
        use del_geo_core::mat3_col_major::Mat3ColMajor;
        use del_geo_core::vec3::Vec3;
        let ddw = {
            let mut ddw = [0f64; 81];
            for (i, j, k, l) in itertools::iproduct!(0..3, 0..3, 0..3, 0..3) {
                let a = dds0[(i + 3 * j) * 9 + (k + 3 * l)][0] * dw0ds[0]
                    + dds0[(i + 3 * j) * 9 + (k + 3 * l)][1] * dw0ds[1]
                    + dds0[(i + 3 * j) * 9 + (k + 3 * l)][2] * dw0ds[2];
                let b = ddw0ds.mult_vec(&ds0[i + 3 * j]).dot(&ds0[k + 3 * l]);
                ddw[(i + 3 * j) * 9 + (k + 3 * l)] = a + b;
            }
            ddw
        };
        let dw0: [f64; 9] =
            std::array::from_fn(|i| ds0[i][0] * dw0ds[0] + ds0[i][1] * dw0ds[1] + ds0[i][2] * dw0ds[2]);
        let eps = 1.0e-4;
        for (k, l) in itertools::iproduct!(0..3, 0..3) {
            let f1 = {
                let mut f1 = f0;
                f1[k + 3 * l] += eps;
                f1
            };
            let (u1, s1, v1) = del_geo_core::mat3_col_major::svd(
                &f1,
                del_geo_core::mat3_sym::EigenDecompositionModes::JacobiNumIter(100),
            )
                .unwrap();
            let (ds1, _dds0) = del_geo_core::mat3_col_major::gradient_and_hessian_of_svd_scale(&u1, &s1, &v1);
            let (w1, dw1ds, _ddw1ds) = wdwddw_energy_density_from_scale_arap(&s1);
            {
                let dw_num = (w1 - w0) / eps;
                let dw_ana = dw0[k + 3 * l];
                println!("## {} {} {} {}", k, l, dw_num, dw_ana);
                assert!((dw_num-dw_ana).abs()<7.0e-4*(dw_ana.abs()+1.0));
            }
            let dw1: [f64; 9] = std::array::from_fn(|i| {
                ds1[i][0] * dw1ds[0] + ds1[i][1] * dw1ds[1] + ds1[i][2] * dw1ds[2]
            });
            for (i, j) in itertools::iproduct!(0..3, 0..3) {
                let ddw_num = (dw1[i + 3 * j] - dw0[i + 3 * j]) / eps;
                let ddw_ana = ddw[(i + 3 * j) * 9 + (k + 3 * l)];
                println!("{} {} {} {} {} {}", i, j, k, l, ddw_num, ddw_ana);
                assert!((ddw_num-ddw_ana).abs()<7.0e-3*(ddw_ana.abs()+1.0));
            }
        }
    }
    Ok(())
}
