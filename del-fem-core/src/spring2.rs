/// returns dp0 and dp1
pub fn pbd(
    p0_def: &[f32;2],
    p1_def: &[f32;2],
    len_ini: f32,
    w0: f32,
    w1: f32)  -> ([f32;2], [f32;2])
{
    let len_def = del_geo_core::edge2::length(p0_def, p1_def);
    let e01 = del_geo_core::edge2::unit_edge_vector(p0_def, p1_def);
    let dp0 = [
        w0 / (w0+w1) * (len_def - len_ini) * e01[0],
        w0 / (w0+w1) * (len_def - len_ini) * e01[1]];
    let dp1 = [
        -w1 / (w0+w1) * (len_def - len_ini) * e01[0],
        -w1 / (w0+w1) * (len_def - len_ini) * e01[1]];
    (dp0,dp1)
}