use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray3, PyUntypedArrayMethods,
};
use pyo3::Python;

mod mesh_laplacian;

#[pyo3::pyfunction]
fn merge_mitc3_to_bsr_for_meshtri2<'a>(
    _py: Python<'a>,
    thick: f32,
    lambda: f32,
    myu: f32,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xy: PyReadonlyArray2<'a, f32>,
    row2idx: PyReadonlyArray1<'a, usize>,
    idx2col: PyReadonlyArray1<'a, usize>,
    mut idx2val: PyReadwriteArray3<'a, f32>,
) {
    let num_vtx = vtx2xy.shape()[0];
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xy = vtx2xy.as_slice().unwrap();
    let row2idx = row2idx.as_slice().unwrap();
    let idx2col = idx2col.as_slice().unwrap();
    let idx2val = idx2val.as_slice_mut().unwrap();
    let mut buffer = vec![usize::MAX, num_vtx];
    for node2vtx in tri2vtx.chunks(3) {
        let node2vtx = arrayref::array_ref![node2vtx, 0, 3];
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = arrayref::array_ref![vtx2xy, i0 * 2, 2];
        let p1 = arrayref::array_ref![vtx2xy, i1 * 2, 2];
        let p2 = arrayref::array_ref![vtx2xy, i2 * 2, 2];
        let (_w, _dw, ddw) = del_fem_core::mitc_tri3::w_dw_ddw_plate_bending(
            &[p0, p1, p2],
            &[[0.; 3]; 3],
            thick,
            lambda,
            myu,
        );
        del_fem_core::merge::blkcsr::<f32, 9, 3>(
            node2vtx,
            node2vtx,
            &ddw,
            row2idx,
            idx2col,
            idx2val,
            &mut buffer,
        );
    }
}

#[pyo3::pyfunction]
fn mitc3_mass_for_trimesh2<'a>(
    _py: Python<'a>,
    thick: f32,
    rho: f32,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xy: PyReadonlyArray2<'a, f32>,
    mut vtx2mass: PyReadwriteArray1<'a, f32>,
) {
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xy = vtx2xy.as_slice().unwrap();
    let vtx2mass = vtx2mass.as_slice_mut().unwrap();
    del_fem_core::mitc_tri3::mass_lumped_plate_bending(tri2vtx, vtx2xy, thick, rho, vtx2mass);
}

#[pyo3::pyfunction]
fn merge_laplace_to_bsr_for_meshtri2<'a>(
    _py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xy: PyReadonlyArray2<'a, f32>,
    row2idx: PyReadonlyArray1<'a, usize>,
    idx2col: PyReadonlyArray1<'a, usize>,
    mut idx2val: PyReadwriteArray1<'a, f32>,
) {
    let num_vtx = vtx2xy.shape()[0];
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xy = vtx2xy.as_slice().unwrap();
    let row2idx = row2idx.as_slice().unwrap();
    let idx2col = idx2col.as_slice().unwrap();
    let idx2val = idx2val.as_slice_mut().unwrap();
    let mut buffer = vec![usize::MAX, num_vtx];
    for node2vtx in tri2vtx.chunks(3) {
        let node2vtx = arrayref::array_ref!(node2vtx, 0, 3);
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = arrayref::array_ref!(vtx2xy, i0 * 2, 2);
        let p1 = arrayref::array_ref!(vtx2xy, i1 * 2, 2);
        let p2 = arrayref::array_ref!(vtx2xy, i2 * 2, 2);
        let ddw = del_fem_core::laplace_tri2::ddw_(1.0, p0, p1, p2);
        del_fem_core::merge::blkcsr::<f32, 1, 3>(
            node2vtx,
            node2vtx,
            &ddw,
            row2idx,
            idx2col,
            idx2val,
            &mut buffer,
        );
    }
}

#[pyo3::pyfunction]
fn merge_laplace_to_bsr_for_meshtri3<'a>(
    _py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f32>,
    row2idx: PyReadonlyArray1<'a, usize>,
    idx2col: PyReadonlyArray1<'a, usize>,
    mut idx2val: PyReadwriteArray1<'a, f32>,
) {
    let num_vtx = vtx2xyz.shape()[0];
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let row2idx = row2idx.as_slice().unwrap();
    let idx2col = idx2col.as_slice().unwrap();
    let idx2val = idx2val.as_slice_mut().unwrap();
    let mut buffer = vec![usize::MAX, num_vtx];
    for node2vtx in tri2vtx.chunks(3) {
        let node2vtx = arrayref::array_ref!(node2vtx, 0, 3);
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = arrayref::array_ref!(vtx2xyz, i0 * 3, 3);
        let p1 = arrayref::array_ref!(vtx2xyz, i1 * 3, 3);
        let p2 = arrayref::array_ref!(vtx2xyz, i2 * 3, 3);
        let ddw = del_geo_core::tri3::emat_cotangent_laplacian(p0, p1, p2);
        del_fem_core::merge::blkcsr::<f32, 1, 3>(
            node2vtx,
            node2vtx,
            &ddw,
            row2idx,
            idx2col,
            idx2val,
            &mut buffer,
        );
    }
}

#[pyo3::pyfunction]
fn merge_linear_solid_to_bsr_for_meshtri2<'a>(
    _py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xy: PyReadonlyArray2<'a, f32>,
    row2idx: PyReadonlyArray1<'a, usize>,
    idx2col: PyReadonlyArray1<'a, usize>,
    mut idx2val: numpy::PyReadwriteArray3<'a, f32>,
) {
    let num_vtx = vtx2xy.shape()[0];
    assert_eq!(idx2val.shape(), [idx2col.shape()[0], 2, 2]);
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xy = vtx2xy.as_slice().unwrap();
    let row2idx = row2idx.as_slice().unwrap();
    let idx2col = idx2col.as_slice().unwrap();
    let idx2val = idx2val.as_slice_mut().unwrap();
    let mut buffer = vec![usize::MAX, num_vtx];
    for node2vtx in tri2vtx.chunks(3) {
        let node2vtx = arrayref::array_ref!(node2vtx, 0, 3);
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = arrayref::array_ref!(vtx2xy, i0 * 2, 2);
        let p1 = arrayref::array_ref!(vtx2xy, i1 * 2, 2);
        let p2 = arrayref::array_ref!(vtx2xy, i2 * 2, 2);
        let ddw = del_fem_core::linear_solid_tri2::emat_tri2(1.0, 1.0, p0, p1, p2);
        del_fem_core::merge::blkcsr::<f32, 4, 3>(
            node2vtx,
            node2vtx,
            &ddw,
            row2idx,
            idx2col,
            idx2val,
            &mut buffer,
        );
    }
}

/// A Python module implemented in Rust.
#[pyo3::pymodule]
#[pyo3(name = "del_fem")]
fn del_fem_(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    mesh_laplacian::add_functions(_py, m)?;
    m.add_function(pyo3::wrap_pyfunction!(mitc3_mass_for_trimesh2, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(merge_mitc3_to_bsr_for_meshtri2, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        merge_laplace_to_bsr_for_meshtri2,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        merge_laplace_to_bsr_for_meshtri3,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        merge_linear_solid_to_bsr_for_meshtri2,
        m
    )?)?;
    Ok(())
}
