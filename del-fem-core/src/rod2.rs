pub fn pbd(
    p0:&[f32;2],
    p1:&[f32;2],
    p2:&[f32;2],
    w: &[f32;3],
    angle_trg: f32) -> [[f32;2];3]
{
    let v01 = del_geo_core::vec2::sub(p1,p0);
    let v12 = del_geo_core::vec2::sub(p2,p1);
    let (t,dt) = del_geo_core::vec2::wdw_angle_between_two_vecs(&v01, &v12);
    let dtdp0 = [-dt[0][0],-dt[0][1]];
    let dtdp1 = [dt[0][0]-dt[1][0],dt[0][1]-dt[1][1]];
    let dtdp2 = [dt[1][0],dt[1][1]];
    let c = t - angle_trg;
    let s = w[0] * del_geo_core::vec2::squared_length(&dtdp0) +
            w[1] * del_geo_core::vec2::squared_length(&dtdp1) +
            w[2] * del_geo_core::vec2::squared_length(&dtdp2);
    let s = c / s;
    [
        [ -s * w[0] * dtdp0[0], -s * w[0] * dtdp0[1]],
        [ -s * w[1] * dtdp1[0], -s * w[1] * dtdp1[1]],
        [ -s * w[2] * dtdp2[0], -s * w[2] * dtdp2[1]]]

}