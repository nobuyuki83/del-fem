pub fn add_weighted_emat_2d<T, const NNO: usize>(
    emat: &mut [[[T; 4]; NNO]; NNO],
    lambda: T,
    myu: T,
    dldx: [[T; NNO]; 2],
    w: T,
) where
    T: num_traits::Float + std::ops::AddAssign,
{
    for (ino, jno) in itertools::iproduct!(0..NNO, 0..NNO) {
        emat[ino][jno][0] += w * (lambda + myu) * dldx[0][ino] * dldx[0][jno];
        emat[ino][jno][1] +=
            w * (lambda * dldx[0][ino] * dldx[1][jno] + myu * dldx[0][jno] * dldx[1][ino]);
        emat[ino][jno][2] +=
            w * (lambda * dldx[1][ino] * dldx[0][jno] + myu * dldx[1][jno] * dldx[0][ino]);
        emat[ino][jno][3] += w * (lambda + myu) * dldx[1][ino] * dldx[1][jno];
        let dtmp1 = w * myu * (dldx[1][ino] * dldx[1][jno] + dldx[0][ino] * dldx[0][jno]);
        emat[ino][jno][0] += dtmp1;
        emat[ino][jno][3] += dtmp1;
    }
}

pub fn emat_tri2<T>(lambda: T, myu: T, p0: &[T; 2], p1: &[T; 2], p2: &[T; 2]) -> [[[T; 4]; 3]; 3]
where
    T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
    f64: num_traits::AsPrimitive<T>,
{
    let area = del_geo_core::tri2::area(p0, p1, p2);
    let (dldx, _) = del_geo_core::tri2::dldx(p0, p1, p2);
    let mut emat = [[[T::zero(); 4]; 3]; 3];
    add_weighted_emat_2d::<T, 3>(&mut emat, lambda, myu, dldx, area);
    emat
}
