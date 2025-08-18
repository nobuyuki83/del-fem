use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::PyModuleMethods;
use pyo3::Python;

pub fn add_functions(
    _py: pyo3::Python,
    m: &pyo3::Bound<pyo3::types::PyModule>,
) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(add_wdwddw_rod3_darboux, m)?)?;
    m.add_function(wrap_pyfunction!(rod3_darboux_update_solution_hair, m)?)?;
    m.add_function(wrap_pyfunction!(
        rod3_darboux_initialize_with_perturbation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        rod3_darboux_orthonormalize_framex_for_hair,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn add_wdwddw_rod3_darboux<'a>(
    _py: Python<'a>,
    vtx2xyz_ini: PyReadonlyArray2<'a, f32>,
    vtx2framex_ini: PyReadonlyArray2<'a, f32>,
    vtx2xyz_def: PyReadonlyArray2<'a, f32>,
    vtx2framex_def: PyReadonlyArray2<'a, f32>,
    mdtt: f32,
    //
    mut w: numpy::PyReadwriteArray0<'a, f32>,
    mut dw: numpy::PyReadwriteArray2<'a, f32>,
    //
    row2idx: PyReadonlyArray1<'a, usize>,
    idx2col: PyReadonlyArray1<'a, usize>,
    mut row2val: numpy::PyReadwriteArray2<'a, f32>,
    mut idx2val: numpy::PyReadwriteArray2<'a, f32>,
) {
    assert!(vtx2xyz_ini.is_c_contiguous());
    assert!(vtx2framex_ini.is_c_contiguous());
    assert!(vtx2xyz_def.is_c_contiguous());
    assert!(vtx2framex_def.is_c_contiguous());
    assert!(dw.is_c_contiguous());
    assert!(row2val.is_c_contiguous());
    assert!(idx2val.is_c_contiguous());
    //
    let num_vtx = vtx2xyz_ini.shape()[0];
    assert_eq!(vtx2xyz_ini.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2framex_ini.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2xyz_def.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2framex_def.shape(), &[num_vtx, 3]);
    assert_eq!(dw.shape(), &[num_vtx, 4]);
    assert_eq!(row2val.shape(), &[num_vtx, 16]);
    //
    use slice_of_array::SliceNestExt;
    let dw: &mut [[f32; 4]] = dw.as_slice_mut().unwrap().nest_mut();
    let ddw = del_fem_cpu::sparse_square::MatrixRefMut::<f32, 16> {
        num_blk: row2idx.len(),
        row2idx: row2idx.as_slice().unwrap(),
        idx2col: idx2col.as_slice().unwrap(),
        idx2val: idx2val.as_slice_mut().unwrap().nest_mut(),
        row2val: row2val.as_slice_mut().unwrap().nest_mut(),
    };
    del_fem_cpu::rod3_darboux::wdwddw_hair_system(
        &mut w.as_slice_mut().unwrap()[0],
        dw,
        ddw,
        vtx2xyz_ini.as_slice().unwrap(),
        vtx2xyz_def.as_slice().unwrap(),
        1.0,
        &[1.0, 1.0, 1.0],
        vtx2framex_ini.as_slice().unwrap(),
        vtx2framex_def.as_slice().unwrap(),
        mdtt,
    );
}

#[pyo3::pyfunction]
fn rod3_darboux_initialize_with_perturbation(
    _py: Python,
    mut vtx2xyz_def: PyReadwriteArray2<f32>,
    mut vtx2framex_def: PyReadwriteArray2<f32>,
    vtx2xyz_ini: PyReadonlyArray2<f32>,
    vtx2framex_ini: PyReadonlyArray2<f32>,
    vtx2isfix: PyReadonlyArray2<i32>,
    pos_mag: f32,
    framex_mag: f32,
) {
    assert!(vtx2xyz_ini.is_c_contiguous());
    assert!(vtx2framex_ini.is_c_contiguous());
    assert!(vtx2xyz_def.is_c_contiguous());
    assert!(vtx2framex_def.is_c_contiguous());
    assert!(vtx2isfix.is_c_contiguous());
    //
    let num_vtx = vtx2xyz_ini.shape()[0];
    assert_eq!(vtx2xyz_ini.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2framex_ini.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2xyz_def.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2framex_def.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2isfix.shape(), &[num_vtx, 4]);
    //
    use rand::SeedableRng;
    let rng = rand_chacha::ChaChaRng::seed_from_u64(0);
    use slice_of_array::SliceNestExt;
    del_fem_cpu::rod3_darboux::initialize_with_perturbation(
        vtx2xyz_def.as_slice_mut().unwrap(),
        vtx2framex_def.as_slice_mut().unwrap(),
        vtx2xyz_ini.as_slice().unwrap(),
        vtx2framex_ini.as_slice().unwrap(),
        vtx2isfix.as_slice().unwrap().nest(),
        pos_mag,
        framex_mag,
        rng,
    );
}

#[pyo3::pyfunction]
fn rod3_darboux_orthonormalize_framex_for_hair(
    _py: Python,
    mut vtx2framex: PyReadwriteArray2<f32>,
    vtx2xyz: PyReadonlyArray2<f32>,
) {
    assert!(vtx2framex.is_c_contiguous());
    assert!(vtx2xyz.is_c_contiguous());
    //
    let num_vtx = vtx2xyz.shape()[0];
    assert_eq!(vtx2xyz.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2framex.shape(), &[num_vtx, 3]);
    del_fem_cpu::rod3_darboux::orthonormalize_framex_for_hair(
        vtx2framex.as_slice_mut().unwrap(),
        vtx2xyz.as_slice().unwrap(),
    );
}

#[pyo3::pyfunction]
fn rod3_darboux_update_solution_hair(
    _py: Python,
    mut vtx2xyz: PyReadwriteArray2<f32>,
    mut vtx2framex: PyReadwriteArray2<f32>,
    vec_x: PyReadonlyArray2<f32>,
    vtx2isfix: PyReadonlyArray2<i32>,
) {
    assert!(vtx2xyz.is_c_contiguous());
    assert!(vtx2framex.is_c_contiguous());
    assert!(vec_x.is_c_contiguous());
    assert!(vtx2isfix.is_c_contiguous());
    //
    let num_vtx = vtx2xyz.shape()[0];
    assert_eq!(vtx2xyz.shape(), &[num_vtx, 3]);
    assert_eq!(vtx2framex.shape(), &[num_vtx, 3]);
    assert_eq!(vec_x.shape(), &[num_vtx, 4]);
    assert_eq!(vtx2isfix.shape(), &[num_vtx, 4]);
    //
    use slice_of_array::SliceNestExt;
    del_fem_cpu::rod3_darboux::update_solution_hair(
        vtx2xyz.as_slice_mut().unwrap(),
        vtx2framex.as_slice_mut().unwrap(),
        vec_x.as_slice().unwrap().nest(),
        vtx2isfix.as_slice().unwrap().nest(),
    );
}
