pub fn merge_from_mesh<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    row2idx: &[usize],
    idx2col: &[usize],
    row2val: &mut [T],
    idx2val: &mut [T],
    merge_buffer: &mut Vec<usize>,
) where
    T: num_traits::Float + 'static + std::ops::AddAssign + std::ops::MulAssign,
    f64: num_traits::AsPrimitive<T>,
{
    for node2vtx in tri2vtx.chunks(3) {
        let node2vtx = arrayref::array_ref![node2vtx, 0, 3];
        let v0: &[T; 3] = arrayref::array_ref!(vtx2xyz, node2vtx[0] * 3, 3);
        let v1: &[T; 3] = arrayref::array_ref!(vtx2xyz, node2vtx[1] * 3, 3);
        let v2: &[T; 3] = arrayref::array_ref!(vtx2xyz, node2vtx[2] * 3, 3);
        let emat: [[[T; 1]; 3]; 3] = del_geo_core::tri3::emat_cotangent_laplacian(v0, v1, v2);
        crate::merge::csrdia::<T, 1, 3>(
            node2vtx,
            node2vtx,
            &emat,
            row2idx,
            idx2col,
            row2val,
            idx2val,
            merge_buffer,
        );
    }
}

/// return linear system
/// I * `val_dia` + L * `val_offdia`
pub fn to_linearsystem(
    tri2vtx: &[usize],
    num_vtx: usize,
    val_dia: f32,
    val_offdia: f32,
) -> del_fem_ls::linearsystem::Solver<f32> {
    let vtx2vtx = del_msh_cpu::vtx2vtx::from_uniform_mesh(tri2vtx, 3, num_vtx, false);
    let mut ls = del_fem_ls::linearsystem::Solver::new();
    ls.initialize(&vtx2vtx.0, &vtx2vtx.1);
    //
    ls.begin_merge();
    let mut buffer = vec![usize::MAX; num_vtx];
    let ddw = del_geo_core::tri3::emat_graph_laplacian(val_offdia);
    for node2vtx in tri2vtx.chunks(3) {
        let node2vtx = arrayref::array_ref!(node2vtx, 0, 3);
        crate::merge::csrdia::<f32, 1, 3>(
            node2vtx,
            node2vtx,
            &ddw,
            &ls.sparse.row2idx,
            &ls.sparse.idx2col,
            &mut ls.sparse.row2val,
            &mut ls.sparse.idx2val,
            &mut buffer,
        );
    }
    ls.sparse.row2val.iter_mut().for_each(|v| *v += val_dia);
    ls.end_merge();
    ls
}
