use del_geo_core::mat3_col_major::Mat3ColMajor;

/// # Argument
/// * `frame` - `frame[i][j]` is `frame[i]` is the `i`-th axis and `j`-th coordinate
/// # Return
/// * `dfdv` - differentiation of frame w.r.t vertex position (i.e., `frame[2] * len`) )
///   * `dfdv[i][j][k]` - differentiation of `frame[i][j]` w.r.t vertex position `v[k]`
fn diff_rod_frame(length: f64, frame: &[[f64; 3]; 3]) -> ([[[f64; 3]; 3]; 3], [[f64; 3]; 3]) {
    use del_geo_core::mat3_col_major;
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    use del_geo_core::vec3::Vec3;
    let dfdv = [
        mat3_col_major::from_scaled_outer_product(1.0 / length, &frame[2], &frame[0])
            .to_mat3_array_of_array(),
        mat3_col_major::from_scaled_outer_product(1.0 / length, &frame[2], &frame[1])
            .to_mat3_array_of_array(),
        mat3_col_major::from_projection_onto_plane(&frame[2])
            .scale(1.0 / length)
            .to_mat3_array_of_array(),
    ];
    let dfdt = [frame[1].scale(-1.0), frame[0], [0f64; 3]];
    (dfdv, dfdt)
}

/// frame after small vertex movement and z-axis rotation
fn updated_rod_frame<T>(ux0: &[T; 3], z0: &[T; 3], dz: &[T; 3], dtheta: T) -> [[T; 3]; 3]
where
    T: num_traits::Float + std::fmt::Display,
{
    use del_geo_core::vec3::Vec3;
    let uz0 = z0.normalize();
    let uy0 = uz0.cross(ux0);
    let uz1 = z0.add(&dz).normalize();
    let r_mat = del_geo_core::mat3_col_major::minimum_rotation_matrix(&uz0, &uz1);
    let ux1 = ux0.scale(dtheta.cos()).sub(&uy0.scale(dtheta.sin()));
    let uy1 = uy0.scale(dtheta.cos()).add(&ux0.scale(dtheta.sin()));
    let ux2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &ux1);
    let uy2 = del_geo_core::mat3_col_major::mult_vec(&r_mat, &uy1);
    [ux2, uy2, uz1]
}

#[test]
fn test_dwddw_rod_frame_trans() {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
    let eps = 1.0e-5f64;
    for itr in 0..100 {
        use del_geo_core::vec3::Vec3;
        let ez = del_geo_core::sphere::sample_surface_uniform::<f64>(&[rng.random(), rng.random()]);
        let len = rng.random::<f64>() + 0.1;
        let (ex, ey) = del_geo_core::vec3::basis_xy_from_basis_z(&ez);
        let frm0 = [ex, ey, ez];
        let q: [f64; 3] = [rng.random(), rng.random(), rng.random()];
        let w0 = [q.dot(&frm0[0]), q.dot(&frm0[1]), q.dot(&frm0[2])];
        let (dw0dv, dw0dt) = {
            let (dfdv, dfdt) = diff_rod_frame(len, &frm0);
            let dwdv = [
                del_geo_core::vec3::mult_mat3_array_of_array(&q, &dfdv[0]),
                del_geo_core::vec3::mult_mat3_array_of_array(&q, &dfdv[1]),
                del_geo_core::vec3::mult_mat3_array_of_array(&q, &dfdv[2]),
            ];
            let dwdt = [q.dot(&dfdt[0]), q.dot(&dfdt[1]), q.dot(&dfdt[2])];
            (dwdv, dwdt)
        };
        let du = [
            2.0 * rng.random::<f64>() - 1.0,
            2.0 * rng.random::<f64>() - 1.0,
            2.0 * rng.random::<f64>() - 1.0,
        ]
        .scale(eps);
        let dt = (2.0 * rng.random::<f64>() - 1.0) * eps;
        let frm1 = updated_rod_frame(&frm0[0], &frm0[2].scale(len), &du, dt);
        let w1 = [q.dot(&frm1[0]), q.dot(&frm1[1]), q.dot(&frm1[2])];
        for i in 0..3 {
            let a = (w1[i] - w0[i]) / eps;
            let b = (dw0dv[i].dot(&du) + dw0dt[i] * dt) / eps;
            let err = (a-b).abs()/(a.abs()+b.abs());
            assert!(err<5.0e-2, "{} {} {} {}", i, a, b, err);
        }
    }
}
