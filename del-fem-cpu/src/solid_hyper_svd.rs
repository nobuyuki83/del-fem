pub trait EnergyDensityFromSingularValueOfDefGrad<Real> {
    fn eval(&self, s: &[Real; 3]) -> Real;
    fn grad(&self, s: &[Real; 3]) -> [Real; 3];
    fn hessian(&self, s: &[Real; 3]) -> [Real; 9];
}

pub struct Arap {}

impl<Real> EnergyDensityFromSingularValueOfDefGrad<Real> for Arap
where
    Real: num_traits::Float,
{
    fn eval(&self, s: &[Real; 3]) -> Real {
        let one = Real::one();
        (s[0] - one).powi(2) + (s[1] - one).powi(2) + (s[2] - one).powi(2)
    }

    fn grad(&self, s: &[Real; 3]) -> [Real; 3] {
        let one = Real::one();
        let two = one + one;
        [two * (s[0] - one), two * (s[1] - one), two * (s[2] - one)]
    }

    fn hessian(&self, _s: &[Real; 3]) -> [Real; 9] {
        let one = Real::one();
        let two = one + one;
        del_geo_core::mat3_col_major::from_diagonal(&[two, two, two])
    }
}

pub struct SymDirichlet {}

impl<Real> EnergyDensityFromSingularValueOfDefGrad<Real> for SymDirichlet
where
    Real: num_traits::Float,
{
    fn eval(&self, s: &[Real; 3]) -> Real {
        let one = Real::one();
        let ss = [s[0] * s[0], s[1] * s[1], s[2] * s[2]];
        ss[0] + one / ss[0] + ss[1] + one / ss[1] + ss[2] + one / ss[2]
    }

    fn grad(&self, s: &[Real; 3]) -> [Real; 3] {
        let one = Real::one();
        let two = one + one;
        let sss = [s[0].powi(3), s[1].powi(3), s[2].powi(3)];
        [
            two * s[0] - two / sss[0],
            two * s[1] - two / sss[1],
            two * s[2] - two / sss[2],
        ]
    }

    fn hessian(&self, s: &[Real; 3]) -> [Real; 9] {
        let zero = Real::zero();
        let one = Real::one();
        let two = one + one;
        let three = two + one;
        let six = two * three;
        let ssss = [s[0].powi(4), s[1].powi(4), s[2].powi(4)];
        [
            two + six / ssss[0],
            zero,
            zero,
            zero,
            two + six / ssss[1],
            zero,
            zero,
            zero,
            two + six / ssss[2],
        ]
    }
}

pub struct Mips {}

impl<Real> EnergyDensityFromSingularValueOfDefGrad<Real> for Mips
where
    Real: num_traits::Float,
{
    fn eval(&self, s: &[Real; 3]) -> Real {
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) / (s[0] * s[1] * s[2])
    }

    fn grad(&self, s: &[Real; 3]) -> [Real; 3] {
        let one = Real::one();
        [
            one / (s[1] * s[2]) - (s[1] / s[2] + s[2] / s[1]) / s[0].powi(2),
            one / (s[2] * s[0]) - (s[2] / s[0] + s[0] / s[2]) / s[1].powi(2),
            one / (s[0] * s[1]) - (s[0] / s[1] + s[1] / s[0]) / s[2].powi(2),
        ]
    }

    fn hessian(&self, s: &[Real; 3]) -> [Real; 9] {
        let one = Real::one();
        let two = one + one;
        let ss = [s[0] * s[0], s[1] * s[1], s[2] * s[2]];
        let t0 = s[0] / (ss[1] * ss[2]) - (one / ss[2] + one / ss[1]) / s[0];
        let t1 = s[1] / (ss[2] * ss[0]) - (one / ss[0] + one / ss[2]) / s[1];
        let t2 = s[2] / (ss[0] * ss[1]) - (one / ss[1] + one / ss[0]) / s[2];
        [
            (s[1] / s[2] + s[2] / s[1]) * two / s[0].powi(3),
            t2,
            t1,
            t2,
            (s[2] / s[0] + s[0] / s[2]) * two / s[1].powi(3),
            t0,
            t1,
            t0,
            (s[0] / s[1] + s[1] / s[0]) * two / s[2].powi(3), // 22
        ]
    }
}

pub struct Ogden {}

impl<Real> EnergyDensityFromSingularValueOfDefGrad<Real> for Ogden
where
    Real: num_traits::Float,
{
    fn eval(&self, s: &[Real; 3]) -> Real {
        let one = Real::one();
        let two = one + one;
        let half = one / two;
        let three = one + two;
        (0..5)
            .map(|k| {
                s[0].powf(half.powi(k)) + s[1].powf(half.powi(k)) + s[2].powf(half.powi(k)) - three
            })
            .fold(Real::zero(), |sum, x| sum + x)
    }

    fn grad(&self, s: &[Real; 3]) -> [Real; 3] {
        let one = Real::one();
        let two = one + one;
        let half = one / two;
        [
            (0..5)
                .map(|i| s[0].powf(half.powi(i) - one) / two.powi(i))
                .fold(Real::zero(), |sum, x| sum + x),
            (0..5)
                .map(|i| s[1].powf(half.powi(i) - one) / two.powi(i))
                .fold(Real::zero(), |sum, x| sum + x),
            (0..5)
                .map(|i| s[2].powf(half.powi(i) - one) / two.powi(i))
                .fold(Real::zero(), |sum, x| sum + x),
        ]
    }

    fn hessian(&self, s: &[Real; 3]) -> [Real; 9] {
        let zero = Real::zero();
        let one = Real::one();
        let two = one + one;
        let half = one / two;
        let t0 = (0..5)
            .map(|i| s[0].powf(half.powi(i) - two) / two.powi(i) * (half.powi(i) - one))
            .fold(Real::zero(), |sum, x| sum + x);
        let t1 = (0..5)
            .map(|i| s[1].powf(half.powi(i) - two) / two.powi(i) * (half.powi(i) - one))
            .fold(Real::zero(), |sum, x| sum + x);
        let t2 = (0..5)
            .map(|i| s[2].powf(half.powi(i) - two) / two.powi(i) * (half.powi(i) - one))
            .fold(Real::zero(), |sum, x| sum + x);
        [t0, zero, zero, zero, t1, zero, zero, zero, t2]
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    fn test_gradient<
        Model: crate::solid_hyper_svd::EnergyDensityFromSingularValueOfDefGrad<f64>,
    >(
        model: Model,
        s0: &[f64; 3],
    ) {
        let w0 = model.eval(&s0);
        let dw0 = model.grad(&s0);
        let ddw0 = model.hessian(&s0);
        let eps = 1.0e-5;
        for j in 0..3 {
            let s1 = {
                let mut s1 = *s0;
                s1[j] += eps;
                s1
            };
            let w1 = model.eval(&s1);
            let dw1 = model.grad(&s1);
            {
                let dw_ana = dw0[j];
                let dw_num = (w1 - w0) / eps;
                assert!(
                    (dw_ana - dw_num).abs() < 1.0e-3 * (dw_ana.abs() + 1.0),
                    "{} --> {} {}",
                    j,
                    dw_ana,
                    dw_num
                );
            }
            for i in 0..3 {
                let dw_ana = ddw0[i + 3 * j];
                let dw_num = (dw1[i] - dw0[i]) / eps;
                assert!(
                    (dw_ana - dw_num).abs() < 1.0e-3 * (dw_ana.abs() + 1.0),
                    "{} --> {} {}",
                    j,
                    dw_ana,
                    dw_num
                );
            }
        }
    }

    #[test]
    fn gradient() {
        use crate::solid_hyper_svd::{Arap, Mips, Ogden, SymDirichlet};
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        let s0 = [rng.random(), rng.random(), rng.random()];
        test_gradient(Arap {}, &s0);
        test_gradient(SymDirichlet {}, &s0);
        test_gradient(Mips {}, &s0);
        test_gradient(Ogden {}, &s0);
    }
}

pub fn energy_density_gradient_hessian_wrt_def_grad<
    Real,
    Model: EnergyDensityFromSingularValueOfDefGrad<Real>,
>(
    f0: &[Real; 9],
    model: &Model,
) -> (Real, [Real; 9], [Real; 81])
where
    Real: num_traits::Float + num_traits::FloatConst,
{
    let (u0, s0, v0) = del_geo_core::mat3_col_major::svd(
        &f0,
        del_geo_core::mat3_sym::EigenDecompositionModes::JacobiNumIter(100),
    )
    .unwrap();
    let (w0, dw0ds, ddw0ds) = (model.eval(&s0), model.grad(&s0), model.hessian(&s0));
    let (ds0, dds0) =
        del_geo_core::mat3_col_major::gradient_and_hessian_of_svd_scale(&u0, &s0, &v0);
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    use del_geo_core::vec3::Vec3;
    let dw0: [Real; 9] =
        std::array::from_fn(|i| ds0[i][0] * dw0ds[0] + ds0[i][1] * dw0ds[1] + ds0[i][2] * dw0ds[2]);
    let ddw = {
        let mut ddw = [Real::zero(); 81];
        for (i, j, k, l) in itertools::iproduct!(0..3, 0..3, 0..3, 0..3) {
            let a = dds0[(i + 3 * j) * 9 + (k + 3 * l)][0] * dw0ds[0]
                + dds0[(i + 3 * j) * 9 + (k + 3 * l)][1] * dw0ds[1]
                + dds0[(i + 3 * j) * 9 + (k + 3 * l)][2] * dw0ds[2];
            let b = ddw0ds.mult_vec(&ds0[i + 3 * j]).dot(&ds0[k + 3 * l]);
            ddw[(i + 3 * j) * 9 + (k + 3 * l)] = a + b;
        }
        ddw
    };
    (w0, dw0, ddw)
}

#[test]
fn test_energy_density_from_singular_value_of_def_grad() -> anyhow::Result<()> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let model = Arap {};
    for _iter in 0..100 {
        let f0: [f64; 9] = std::array::from_fn(|_| rng.random::<f64>());
        let (w0, dw0, ddw) = energy_density_gradient_hessian_wrt_def_grad(&f0, &model);
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
            let (ds1, _dds0) =
                del_geo_core::mat3_col_major::gradient_and_hessian_of_svd_scale(&u1, &s1, &v1);
            let (w1, dw1ds, _ddw1ds) = (model.eval(&s1), model.grad(&s1), model.hessian(&s1));
            {
                let dw_num = (w1 - w0) / eps;
                let dw_ana = dw0[k + 3 * l];
                println!("## {} {} {} {}", k, l, dw_num, dw_ana);
                assert!((dw_num - dw_ana).abs() < 7.0e-4 * (dw_ana.abs() + 1.0));
            }
            let dw1: [f64; 9] = std::array::from_fn(|i| {
                ds1[i][0] * dw1ds[0] + ds1[i][1] * dw1ds[1] + ds1[i][2] * dw1ds[2]
            });
            for (i, j) in itertools::iproduct!(0..3, 0..3) {
                let ddw_num = (dw1[i + 3 * j] - dw0[i + 3 * j]) / eps;
                let ddw_ana = ddw[(i + 3 * j) * 9 + (k + 3 * l)];
                println!("{} {} {} {} {} {}", i, j, k, l, ddw_num, ddw_ana);
                assert!((ddw_num - ddw_ana).abs() < 7.0e-3 * (ddw_ana.abs() + 1.0));
            }
        }
    }
    Ok(())
}
