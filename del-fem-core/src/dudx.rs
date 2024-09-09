pub fn def_grad_tensor<const NDIM: usize, Real>(dudx: &[[Real; NDIM]; NDIM]) -> [[Real; NDIM]; NDIM]
where
    Real: num_traits::Float + std::ops::AddAssign,
{
    let one = Real::one();
    let mut z = [[Real::zero(); NDIM]; NDIM];
    for idim in 0..NDIM {
        for jdim in 0..NDIM {
            z[idim][jdim] = dudx[idim][jdim];
        }
        z[idim][idim] += one;
    }
    z
}

/// C=F^TF = (Z + I)^T(Z+I) = Z^TZ + Z + Z^T + I
pub fn right_cauchy_green_tensor<const NDIM: usize, Real>(
    dudx: &[[Real; NDIM]; NDIM],
) -> [[Real; NDIM]; NDIM]
where
    Real: num_traits::Float + std::ops::AddAssign,
{
    let mut c = [[Real::zero(); NDIM]; NDIM];
    for idim in 0..NDIM {
        for jdim in 0..NDIM {
            c[idim][jdim] = dudx[idim][jdim] + dudx[jdim][idim];
            for kdim in 0..NDIM {
                c[idim][jdim] += dudx[kdim][idim] * dudx[kdim][jdim];
            }
        }
        c[idim][idim] += Real::one();
    }
    c
}

pub const ISTDIM2IJ: [[usize; 2]; 6] = [[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0]];

pub fn tensor3_from_symmetric_vector_param<Real>(v: &[Real; 6]) -> [[Real; 3]; 3]
where
    Real: num_traits::Float,
{
    [[v[0], v[3], v[5]], [v[3], v[1], v[4]], [v[5], v[4], v[2]]]
}

