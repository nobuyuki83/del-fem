use num_traits::AsPrimitive;

pub fn ddw_<T>(alpha: T, p0: &[T; 2], p1: &[T; 2], p2: &[T; 2]) -> [[[T; 1]; 3]; 3]
where
    T: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<T>,
{
    const N_NODE: usize = 3;
    let area = del_geo_core::tri2::area(p0, p1, p2);
    let (dldx, _) = del_geo_core::tri2::dldx(p0, p1, p2);
    let mut ddw = [[[T::zero(); 1]; N_NODE]; N_NODE];
    for (ino, jno) in itertools::iproduct!(0..3, 0..3) {
        ddw[ino][jno][0] =
            alpha * area * (dldx[0][ino] * dldx[0][jno] + dldx[1][ino] * dldx[1][jno]);
    }
    ddw
}
