use num_traits::AsPrimitive;

pub fn ddw_<T>(
    alpha: T,
    p0: &[T; 2],
    p1: &[T; 2],
    p2: &[T; 2]) -> [[[T; 1];3];3]
    where T: num_traits::Float + Copy + 'static,
          f64: AsPrimitive<T>
{
    const N_NODE: usize = 3;
    let area = del_geo::tri2::area_(p0, p1, p2);
    let (dldx, _) = del_geo::tri2::dldx_(p0, p1, p2);
    let mut ddw = [[[T::zero();1]; N_NODE]; N_NODE];
    for ino in 0..N_NODE {
        for jno in 0..N_NODE {
            ddw[ino][jno][0] = alpha * area * (dldx[0][ino] * dldx[0][jno] + dldx[1][ino] * dldx[1][jno]);
        }
    }
    ddw
}