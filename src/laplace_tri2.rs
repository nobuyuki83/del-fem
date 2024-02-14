use num_traits::AsPrimitive;

fn ddw_<T>(
    alpha: T,
    coords: &[[T;2];3])
where T: num_traits::Float + Copy + 'static,
  f64: AsPrimitive<T>
{
  const N_NODE: usize = 3;
  const N_DIM: usize = 2;
  //
  let area = del_geo::tri2::area_(&coords[0], &coords[1], &coords[2]);
  //
  let dldx = del_geo::tri2::dldx_(&coords[0], &coords[1], &coords[2]);
  //
  let mut ddw = [[T::zero(); N_NODE]; N_NODE];
  for ino in 0..N_NODE {
    for jno in 0..N_NODE {
      ddw[ino][jno] = alpha*area*(dldx[ino][0]*dldx[jno][0]+dldx[ino][1]*dldx[jno][1]);
    }
  }
}