fn hessian(
    myu: f32,
    lambda: f32,
    node2xyz: &[[f32; 3]; 8],
    i_gauss_degree: usize) -> [[[f32; 9]; 8]; 8]
{
    use crate::quadrature_line::Quadrature;
    let quadrature = crate::quadrature_line::Quad::<f32>::hoge(i_gauss_degree);
    let num_quadr = quadrature.len();

    let mut emat = [[[0f32; 9]; 8]; 8];
    for (ir1, ir2, ir3) in itertools::iproduct!(0..num_quadr, 0..num_quadr, 0..num_quadr) {
        let (dndx, detwei) = del_geo_core::hex::grad_shapefunc(node2xyz, quadrature, ir1, ir2, ir3);
        for (ino, jno) in itertools::iproduct!(0..8, 0..8) {
            let mut dtmp1 = 0f32;
            for idim in 0..3 {
                for jdim in 0..3 {
                    emat[ino][jno][idim*3+jdim] += detwei * (lambda * dndx[ino][idim] * dndx[jno][jdim]
                        + myu * dndx[jno][idim] * dndx[ino][jdim]);
                }
                dtmp1 += dndx[ino][idim] * dndx[jno][idim];
            }
            for idim in 0..3 {
                emat[ino][jno][idim*3+idim] += detwei * myu * dtmp1;
            }
        }
    }
    emat
}
