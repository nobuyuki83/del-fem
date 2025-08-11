use numpy::IntoPyArray;
use numpy::{
    PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::{Bound, Python};

pub fn add_functions(
    _py: pyo3::Python,
    m: &pyo3::Bound<pyo3::types::PyModule>,
) -> pyo3::PyResult<()> {
    use pyo3::types::PyModuleMethods;
    m.add_function(pyo3::wrap_pyfunction!(block_sparse_apply_bc, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        block_sparse_set_fixed_bc_to_rhs_vector,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(conjugate_gradient, m)?)?;
    Ok(())
}

#[pyo3::pyfunction]
fn block_sparse_apply_bc(
    _py: Python,
    val_dia: f32,
    blk2isfix: PyReadonlyArray2<i32>,
    mut row2val: PyReadwriteArray2<f32>,
    mut idx2val: PyReadwriteArray2<f32>,
    row2idx: PyReadonlyArray1<usize>,
    idx2col: PyReadonlyArray1<usize>,
) {
    assert!(blk2isfix.is_c_contiguous());
    assert!(row2val.is_c_contiguous());
    assert!(idx2val.is_c_contiguous());
    assert!(row2idx.is_c_contiguous());
    assert!(idx2col.is_c_contiguous());
    //
    use slice_of_array::SliceNestExt;
    let num_dim = blk2isfix.shape()[1];
    let num_blk = blk2isfix.shape()[0];
    let num_idx = idx2val.shape()[0];
    if num_dim == 4 {
        assert_eq!(row2val.shape(), &[num_blk, 16]);
        assert_eq!(idx2val.shape(), &[num_idx, 16]);
        assert_eq!(row2idx.shape()[0], num_blk + 1);
        assert_eq!(idx2col.shape()[0], num_idx);
        del_fem_cpu::sparse_square::set_fixed_bc::<f32, 4, 16>(
            val_dia,
            blk2isfix.as_slice().unwrap().nest(),
            row2val.as_slice_mut().unwrap().nest_mut(),
            idx2val.as_slice_mut().unwrap().nest_mut(),
            row2idx.as_slice().unwrap(),
            idx2col.as_slice().unwrap(),
        );
    } else {
        todo!()
    }
}

#[pyo3::pyfunction]
fn block_sparse_set_fixed_bc_to_rhs_vector(
    _py: Python,
    blk2isfix: PyReadonlyArray2<i32>,
    mut rhs: PyReadwriteArray2<f32>,
) {
    use slice_of_array::SliceNestExt;
    let num_dim = blk2isfix.shape()[1];
    let num_blk = blk2isfix.shape()[0];
    if num_dim == 4 {
        assert_eq!(rhs.shape(), &[num_blk, 4]);
        del_fem_cpu::sparse_square::set_fix_dof_to_rhs_vector::<f32, 4>(
            rhs.as_slice_mut().unwrap().nest_mut(),
            blk2isfix.as_slice().unwrap().nest(),
        );
    }
}

#[pyo3::pyfunction]
fn conjugate_gradient<'a>(
    _py: Python<'a>,
    mut r_vec: PyReadwriteArray2<'a, f32>,
    mut u_vec: PyReadwriteArray2<'a, f32>,
    mut p_vec: PyReadwriteArray2<'a, f32>,
    mut ap_vec: PyReadwriteArray2<'a, f32>,
    row2idx: PyReadonlyArray1<'a, usize>,
    idx2col: PyReadonlyArray1<'a, usize>,
    idx2val: PyReadonlyArray2<'a, f32>,
    row2val: PyReadonlyArray2<'a, f32>,
) -> Bound<'a, PyArray1<f32>> {
    use slice_of_array::SliceNestExt;
    assert!(r_vec.is_c_contiguous());
    assert!(u_vec.is_c_contiguous());
    assert!(p_vec.is_c_contiguous());
    assert!(ap_vec.is_c_contiguous());
    assert!(row2idx.is_c_contiguous());
    assert!(idx2col.is_c_contiguous());
    assert!(idx2val.is_c_contiguous());
    assert!(row2val.is_c_contiguous());
    let num_blk = r_vec.shape()[0];
    let num_dim = r_vec.shape()[1];
    assert_eq!(num_blk + 1, row2idx.shape()[0]);
    assert_eq!(idx2val.shape()[0], idx2col.shape()[0]);
    let ddw = del_fem_cpu::sparse_square::MatrixRef {
        num_blk,
        row2idx: row2idx.as_slice().unwrap(),
        idx2col: idx2col.as_slice().unwrap(),
        idx2val: idx2val.as_slice().unwrap().nest(),
        row2val: row2val.as_slice().unwrap().nest(),
    };
    let hist = if num_dim == 4 {
        assert_eq!(u_vec.shape(), &[num_blk, 4]);
        assert_eq!(ap_vec.shape(), &[num_blk, 4]);
        assert_eq!(p_vec.shape(), &[num_blk, 4]);
        del_fem_cpu::sparse_square::conjugate_gradient::<f32, 4, 16>(
            r_vec.as_slice_mut().unwrap().nest_mut(),
            u_vec.as_slice_mut().unwrap().nest_mut(),
            ap_vec.as_slice_mut().unwrap().nest_mut(),
            p_vec.as_slice_mut().unwrap().nest_mut(),
            1.0e-5,
            1000,
            ddw,
        )
    } else {
        todo!();
    };
    hist.into_pyarray(_py)
}
