pub fn merge_from_mesh<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    row2idx: &[usize],
    idx2col: &[usize],
    row2val: &mut [T],
    idx2val: &mut [T],
    merge_buffer: &mut Vec<usize>,
) where
    T: num_traits::Float + 'static + std::ops::AddAssign,
    f64: num_traits::AsPrimitive<T>,
{
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let v0: &[T; 3] = &vtx2xyz[i0 * 3..i0 * 3 + 3].try_into().unwrap();
        let v1: &[T; 3] = &vtx2xyz[i1 * 3..i1 * 3 + 3].try_into().unwrap();
        let v2: &[T; 3] = &vtx2xyz[i2 * 3..i2 * 3 + 3].try_into().unwrap();
        let emat: [[[T; 1]; 3]; 3] = del_geo::tri3::emat_cotangent_laplacian(v0, v1, v2);
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
