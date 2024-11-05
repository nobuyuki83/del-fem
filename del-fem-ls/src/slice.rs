//!  operations on slice of an array

/// dot product
pub fn dot<T>(v0: &[T], v1: &[T]) -> T
where
    T: Copy + std::ops::Mul<Output = T> + num_traits::Zero,
{
    assert_eq!(v0.len(), v1.len());
    v0.iter()
        .zip(v1.iter())
        .fold(T::zero(), |sum, (&x, &y)| sum + x * y)
}

pub fn add_scaled_vector<T>(u: &mut [T], alpha: T, p: &[T])
where
    T: std::ops::Mul<Output = T> + std::ops::AddAssign + Copy,
{
    assert_eq!(u.len(), p.len());
    u.iter_mut()
        .zip(p.iter())
        .for_each(|(a, &b)| *a += alpha * b);
}

/// {p} = {r} + beta*{p}
pub fn scale_and_add_vec<T>(p: &mut [T], beta: T, r: &[T])
where
    T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy,
{
    assert_eq!(r.len(), p.len());
    for i in 0..p.len() {
        p[i] = r[i] + beta * p[i];
    }
}

pub fn set_zero<T>(p: &mut [T])
where
    T: Copy + num_traits::Zero,
{
    p.iter_mut().for_each(|v| *v = T::zero());
}

pub fn copy<T>(p: &mut [T], u: &[T])
where
    T: Copy,
{
    assert_eq!(p.len(), u.len());
    p.iter_mut().zip(u.iter()).for_each(|(a, &b)| *a = b);
}

pub fn sub(p: &mut [f32], u: &[f32], v: &[f32]) {
    assert_eq!(p.len(), u.len());
    for i in 0..p.len() {
        p[i] = u[i] - v[i];
    }
}
