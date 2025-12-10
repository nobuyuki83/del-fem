pub fn nan_indices<T>(xs: &[T]) -> Vec<usize>
where
    T: num_traits::Float,
{
    xs.iter()
        .enumerate()
        .filter(|(_, x)| x.is_nan())
        .map(|(i, _)| i)
        .collect()
}

pub fn nan_indices_from_array<T, const N: usize>(xs: &[[T; N]]) -> Vec<(usize, usize)>
where
    T: num_traits::Float,
{
    let mut res = Vec::new();
    for (i, row) in xs.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            if val.is_nan() {
                res.push((i, j));
            }
        }
    }
    res
}