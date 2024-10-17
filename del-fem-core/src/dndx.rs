pub fn disp_grad_tensor<const NNO: usize, const NDIM: usize, Real>(
    dndx: &[[Real; NDIM]; NNO],
    node2disp: &[[Real; NDIM]; NNO],
) -> [[Real; NDIM]; NDIM]
where
    Real: num_traits::Float + std::ops::AddAssign,
{
    let zero = Real::zero();
    let mut dudx = [[zero; NDIM]; NDIM];
    for idim in 0..NDIM {
        for jdim in 0..NDIM {
            let mut dtmp1 = zero;
            for ino in 0..NNO {
                dtmp1 += node2disp[ino][idim] * dndx[ino][jdim];
            }
            dudx[idim][jdim] = dtmp1;
        }
    }
    dudx
}
