pub fn wdwddw_squared_length_difference<T>(
    stiffness: T,
    node2xyz_def: &[[T; 3]; 2],
    edge_length_ini: T,
) -> (T, [[T; 3]; 2], [[[T; 9]; 2]; 2])
where
    T: num_traits::Float,
{
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    use del_geo_core::vec3::Vec3;
    //
    let one = T::one();
    let half = one / (one + one);
    let v = node2xyz_def[0].sub(&node2xyz_def[1]);
    let l = v.norm();
    let c = edge_length_ini - l;
    let dw = [v.scale(-c * stiffness / l), v.scale(c * stiffness / l)];
    let m = {
        let mvv = del_geo_core::mat3_col_major::from_scaled_outer_product(one, &v, &v);
        let t0 = stiffness * edge_length_ini / (l * l * l);
        let t1 = stiffness * (l - edge_length_ini) / l;
        let t2 = del_geo_core::mat3_col_major::from_identity().scale(t1);
        mvv.scale(t0).add(&t2)
    };
    let ddw = [[m, m.scale(-one)], [m.scale(-one), m]];
    let w = half * stiffness * c * c;
    (w, dw, ddw)
}
