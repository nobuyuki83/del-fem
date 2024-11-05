//! block sparse square matrix

/// generalized matrix-vector multiplication
/// where matrix is block sparse matrix
/// `{y_vec} <- \alpha * [a_mat] * {x_vec} + \beta * {y_vec}`
pub fn gemv_for_block_sparse_matrix_nalgebra<
    T: nalgebra::RealField + Copy,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    SVECM: nalgebra::StorageMut<T, R, C>,
    SVEC: nalgebra::Storage<T, R, C>,
    SMAT: nalgebra::Storage<T, R, R>,
>(
    y_vec: &mut Vec<nalgebra::Matrix<T, R, C, SVECM>>,
    beta: T,
    alpha: T,
    a_mat: &crate::sparse_square::Matrix<nalgebra::Matrix<T, R, R, SMAT>>,
    x_vec: &Vec<nalgebra::Matrix<T, R, C, SVEC>>,
) where
    f32: num_traits::AsPrimitive<T>,
{
    assert_eq!(y_vec.len(), a_mat.num_blk);
    for m in y_vec.iter_mut() {
        (*m).scale_mut(beta);
    }
    for irow in 0..a_mat.num_blk {
        for idx0 in a_mat.row2idx[irow]..a_mat.row2idx[irow + 1] {
            assert!(idx0 < a_mat.idx2col.len());
            let jcol0 = a_mat.idx2col[idx0];
            assert!(jcol0 < a_mat.num_blk);
            y_vec[irow].gemm(alpha, &a_mat.idx2val[idx0], &x_vec[jcol0], T::one());
            // SIMD?
        }
        y_vec[irow].gemm(alpha, &a_mat.row2val[irow], &x_vec[irow], T::one());
    }
}

#[test]
fn test_block33() {
    type MAT = nalgebra::Matrix3<f32>;
    type VEC = nalgebra::Vector3<f32>;
    let mut sparse = crate::sparse_square::Matrix::<MAT>::new();
    let colind = vec![0, 2, 5, 8, 10];
    let rowptr = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    sparse.symbolic_initialization(&colind, &rowptr);
    sparse.set_zero();
    {
        let emat = [
            nalgebra::Matrix3::<f32>::identity(),
            nalgebra::Matrix3::<f32>::zeros(),
            nalgebra::Matrix3::<f32>::zeros(),
            nalgebra::Matrix3::<f32>::identity(),
        ];
        let mut tmp_buffer = Vec::<usize>::new();
        sparse.merge(&[0, 1], &[0, 1], &emat, &mut tmp_buffer);
    }
    let nblk = colind.len() - 1;
    let mut rhs = Vec::<VEC>::new();
    rhs.resize(nblk, Default::default());
    let mut lhs = Vec::<VEC>::new();
    lhs.resize(nblk, Default::default());
    gemv_for_block_sparse_matrix_nalgebra(&mut lhs, 1.0, 1.0, &sparse, &rhs);
}
