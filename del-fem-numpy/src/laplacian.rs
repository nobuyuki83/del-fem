use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use numpy::{PyReadwriteArray1, PyUntypedArrayMethods};
use pyo3::prelude::PyModuleMethods;
use pyo3::Python;

pub fn add_functions(
    _py: pyo3::Python,
    m: &pyo3::Bound<pyo3::types::PyModule>,
) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    // topology
    m.add_function(wrap_pyfunction!(
        merge_hessian_mesh_laplacian_on_trimesh3,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        merge_laplace_to_bsr_for_meshtri2,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        merge_laplace_to_bsr_for_meshtri3,
        m
    )?)?;
    Ok(())
}

#[pyo3::pyfunction]
pub fn merge_hessian_mesh_laplacian_on_trimesh3<'a>(
    _py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz: PyReadonlyArray2<'a, f64>,
    row2idx: PyReadonlyArray1<'a, usize>,
    idx2col: PyReadonlyArray1<'a, usize>,
    mut row2val: numpy::PyReadwriteArray1<'a, f64>,
    mut idx2val: numpy::PyReadwriteArray1<'a, f64>,
) {
    assert!(tri2vtx.is_c_contiguous());
    assert!(vtx2xyz.is_c_contiguous());
    assert!(row2idx.is_c_contiguous());
    assert!(idx2col.is_c_contiguous());
    assert!(row2val.is_c_contiguous());
    assert!(idx2val.is_c_contiguous());
    let mut merge_buffer = vec![0_usize; 0];
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let row2idx = row2idx.as_slice().unwrap();
    let idx2col = idx2col.as_slice().unwrap();
    let row2val = row2val.as_slice_mut().unwrap();
    let idx2val = idx2val.as_slice_mut().unwrap();
    for node2vtx in tri2vtx.chunks(3) {
        let node2vtx = arrayref::array_ref!(node2vtx, 0, 3);
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let v0 = arrayref::array_ref!(vtx2xyz, i0 * 3, 3);
        let v1 = arrayref::array_ref!(vtx2xyz, i1 * 3, 3);
        let v2 = arrayref::array_ref!(vtx2xyz, i2 * 3, 3);
        let emat = del_geo_core::tri3::emat_cotangent_laplacian(v0, v1, v2);
        del_fem_cpu::merge::csrdia(
            node2vtx,
            node2vtx,
            &emat,
            row2idx,
            idx2col,
            row2val,
            idx2val,
            &mut merge_buffer,
        );
    }
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
        let ddw = del_fem_cpu::laplace_tri2::ddw_(1.0, p0, p1, p2);
        del_fem_cpu::merge::blkcsr::<f32, 1, 3>(
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
        del_fem_cpu::merge::blkcsr::<f32, 1, 3>(
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
