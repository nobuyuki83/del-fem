pub fn wdw<T>(p_ini: &[[T; 3]; 4], p_def: &[[T; 3]; 4]) -> ([T; 3], [[[T; 3]; 4]; 3])
where
    T: num_traits::Float + std::ops::AddAssign,
{
    let two = T::one() + T::one();
    let three = two + T::one();
    let area0 = del_geo_core::tri3::area(&p_ini[0], &p_ini[2], &p_ini[3]);
    let area1 = del_geo_core::tri3::area(&p_ini[1], &p_ini[3], &p_ini[2]);
    let len = del_geo_core::edge::length::<T, 3>(&p_ini[2], &p_ini[3]);
    let h0 = two * area0 / len;
    let h1 = two * area1 / len;
    let e23 = del_geo_core::vec3::sub(&p_ini[3], &p_ini[2]);
    let e02 = del_geo_core::vec3::sub(&p_ini[2], &p_ini[0]);
    let e03 = del_geo_core::vec3::sub(&p_ini[3], &p_ini[0]);
    let e12 = del_geo_core::vec3::sub(&p_ini[2], &p_ini[1]);
    let e13 = del_geo_core::vec3::sub(&p_ini[3], &p_ini[1]);
    let cot023 = -del_geo_core::vec3::dot(&e02, &e23) / h0;
    let cot032 = del_geo_core::vec3::dot(&e03, &e23) / h0;
    let cot123 = -del_geo_core::vec3::dot(&e12, &e23) / h1;
    let cot132 = del_geo_core::vec3::dot(&e13, &e23) / h1;
    let tmp0 = three.sqrt() / ((area0 + area1).sqrt() * len);
    let k = [
        (-cot023 - cot032) * tmp0,
        (-cot123 - cot132) * tmp0,
        (cot032 + cot132) * tmp0,
        (cot023 + cot123) * tmp0,
    ];
    let mut w = [T::zero(); 3];
    w[0] = k[0] * p_def[0][0] + k[1] * p_def[1][0] + k[2] * p_def[2][0] + k[3] * p_def[3][0];
    w[1] = k[0] * p_def[0][1] + k[1] * p_def[1][1] + k[2] * p_def[2][1] + k[3] * p_def[3][1];
    w[2] = k[0] * p_def[0][2] + k[1] * p_def[1][2] + k[2] * p_def[2][2] + k[3] * p_def[3][2];
    let mut dwdp = [[[T::zero(); 3]; 4]; 3];
    for i_dim in 0..3 {
        dwdp[i_dim][0][i_dim] = k[0];
        dwdp[i_dim][1][i_dim] = k[1];
        dwdp[i_dim][2][i_dim] = k[2];
        dwdp[i_dim][3][i_dim] = k[3];
    }
    (w, dwdp)
}

#[test]
fn test() {
    let p_ini = [
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.5],
        [0.3, 0.4, 0.3],
        [0.5, 0.2, 0.4],
    ];
    let p0_def = [
        [0.11, 0.04, 0.38],
        [0.22, 0.13, 0.07],
        [0.03, 0.42, 0.35],
        [0.54, 0.01, 0.46],
    ];
    let (c0, dcdp) = wdw(&p_ini, &p0_def);
    let eps = 1.0e-5f64;
    for (i_no, i_dim) in itertools::iproduct!(0..4, 0..3) {
        let p1_def = {
            let mut p1 = p0_def;
            p1[i_no][i_dim] += eps;
            p1
        };
        let (c1, _) = wdw(&p_ini, &p1_def);
        for j_dim in 0..3 {
            let v_ana = dcdp[j_dim][i_no][i_dim];
            let v_num = (c1[j_dim] - c0[j_dim]) / eps;
            // println!("{} {}", v_num, v_ana);
            assert!((v_ana - v_num).abs() < 1.0e-6f64, "{} {}", v_num, v_ana);
        }
    }
}
