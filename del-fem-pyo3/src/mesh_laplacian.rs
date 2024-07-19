use numpy::PyUntypedArrayMethods;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::Python;

pub fn add_functions(_py: pyo3::Python, m: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    use pyo3::wrap_pyfunction;
    // topology
    m.add_function(wrap_pyfunction!(
        merge_hessian_mesh_laplacian_on_trimesh3,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(optimal_rotations_arap_spoke, m)?)?;
    m.add_function(wrap_pyfunction!(residual_arap_spoke, m)?)?;
    m.add_function(wrap_pyfunction!(
        optimal_rotations_arap_spoke_rim_trimesh3,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(residual_arap_spoke_rim_trimesh3, m)?)?;
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
        del_fem_core::merge::csrdia(
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
pub fn optimal_rotations_arap_spoke<'a>(
    _py: Python<'a>,
    vtx2xyz_ini: PyReadonlyArray2<'a, f64>,
    vtx2xyz_def: PyReadonlyArray2<'a, f64>,
    vtx2idx: PyReadonlyArray1<'a, usize>,
    idx2vtx: PyReadonlyArray1<'a, usize>,
    idx2val: PyReadonlyArray1<'a, f64>,
    mut vtx2rot: numpy::PyReadwriteArray3<'a, f64>,
) {
    assert!(vtx2xyz_ini.is_c_contiguous());
    let num_vtx = vtx2xyz_ini.shape()[0];
    assert_eq!(vtx2xyz_ini.shape(), [num_vtx, 3]);
    assert!(vtx2xyz_def.is_c_contiguous());
    assert_eq!(vtx2xyz_ini.shape(), vtx2xyz_def.shape());
    assert!(vtx2idx.is_c_contiguous());
    assert_eq!(vtx2idx.shape(), [num_vtx + 1]);
    assert!(idx2vtx.is_c_contiguous());
    assert!(idx2val.is_c_contiguous());
    assert!(vtx2rot.is_c_contiguous());
    assert_eq!(vtx2rot.shape(), [num_vtx, 3, 3]);
    let vtx2xyz_ini = vtx2xyz_ini.as_slice().unwrap();
    let vtx2xyz_def = vtx2xyz_def.as_slice().unwrap();
    let vtx2idx = vtx2idx.as_slice().unwrap();
    let idx2col = idx2vtx.as_slice().unwrap();
    let idx2val = idx2val.as_slice().unwrap();
    let vtx2rot = vtx2rot.as_slice_mut().unwrap();
    for i_vtx in 0..num_vtx {
        let adj2vtx = &idx2col[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]];
        let adj2weight = &idx2val[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]];
        let rot = del_fem_core::arap::optimal_rotation_for_arap_spoke(
            i_vtx,
            adj2vtx,
            vtx2xyz_ini,
            vtx2xyz_def,
            adj2weight,
            -1.,
        );
        // transpose to change column-major to row-major
        rot.transpose()
            .iter()
            .enumerate()
            .for_each(|(i, &v)| vtx2rot[i_vtx * 9 + i] = v);
    }
}

#[pyo3::pyfunction]
#[allow(clippy::identity_op)]
pub fn residual_arap_spoke<'a>(
    _py: Python<'a>,
    vtx2xyz_ini: PyReadonlyArray2<'a, f64>,
    vtx2xyz_def: PyReadonlyArray2<'a, f64>,
    vtx2idx: PyReadonlyArray1<'a, usize>,
    idx2vtx: PyReadonlyArray1<'a, usize>,
    idx2val: PyReadonlyArray1<'a, f64>,
    vtx2rot: numpy::PyReadonlyArray3<'a, f64>,
    mut vtx2res: numpy::PyReadwriteArray2<'a, f64>,
) {
    assert!(vtx2xyz_ini.is_c_contiguous());
    let num_vtx = vtx2xyz_ini.shape()[0];
    assert!(vtx2xyz_def.is_c_contiguous());
    assert_eq!(vtx2xyz_ini.shape(), [num_vtx, 3]);
    assert_eq!(vtx2xyz_ini.shape(), vtx2xyz_def.shape());
    assert!(vtx2idx.is_c_contiguous());
    assert!(idx2vtx.is_c_contiguous());
    assert!(idx2val.is_c_contiguous());
    assert!(vtx2rot.is_c_contiguous());
    assert_eq!(vtx2rot.shape(), [num_vtx, 3, 3]);
    let vtx2xyz_ini = vtx2xyz_ini.as_slice().unwrap();
    let vtx2xyz_def = vtx2xyz_def.as_slice().unwrap();
    let vtx2idx = vtx2idx.as_slice().unwrap();
    let idx2col = idx2vtx.as_slice().unwrap();
    let idx2val = idx2val.as_slice().unwrap();
    let vtx2rot = vtx2rot.as_slice().unwrap();
    let vtx2res = vtx2res.as_slice_mut().unwrap();
    vtx2res.fill(0.);
    for i_vtx in 0..num_vtx {
        let r_i = nalgebra::Matrix3::<f64>::from_row_slice(&vtx2rot[i_vtx * 9..i_vtx * 9 + 9]);
        let p_i = del_msh_core::vtx2xyz::to_navec3(vtx2xyz_ini, i_vtx);
        let q_i = del_msh_core::vtx2xyz::to_navec3(vtx2xyz_def, i_vtx);
        let adj2vtx = &idx2col[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]];
        let adj2weight = &idx2val[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]];
        for (&j_vtx, &w) in adj2vtx.iter().zip(adj2weight.iter()) {
            let r_j = nalgebra::Matrix3::<f64>::from_row_slice(&vtx2rot[j_vtx * 9..j_vtx * 9 + 9]);
            let p_j = del_msh_core::vtx2xyz::to_navec3(vtx2xyz_ini, j_vtx);
            let q_j = del_msh_core::vtx2xyz::to_navec3(vtx2xyz_def, j_vtx);
            let r = (q_i - q_j) - (r_i + r_j).scale(0.5) * (p_i - p_j);
            let r = r.scale(w);
            vtx2res[i_vtx * 3 + 0] += r.x;
            vtx2res[i_vtx * 3 + 1] += r.y;
            vtx2res[i_vtx * 3 + 2] += r.z;
        }
    }
}

#[pyo3::pyfunction]
pub fn optimal_rotations_arap_spoke_rim_trimesh3<'a>(
    _py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz_ini: PyReadonlyArray2<'a, f64>,
    vtx2xyz_def: PyReadonlyArray2<'a, f64>,
    mut vtx2rot: numpy::PyReadwriteArray3<'a, f64>,
) {
    assert!(tri2vtx.is_c_contiguous());
    assert_eq!(tri2vtx.shape().len(), 2);
    assert_eq!(tri2vtx.shape()[1], 3);
    assert!(vtx2xyz_ini.is_c_contiguous());
    assert!(vtx2xyz_def.is_c_contiguous());
    assert_eq!(vtx2xyz_ini.shape(), vtx2xyz_def.shape());
    assert_eq!(vtx2xyz_ini.shape().len(), 2);
    assert_eq!(vtx2xyz_ini.shape()[1], 3);
    assert!(vtx2rot.is_c_contiguous());
    assert_eq!(vtx2rot.shape().len(), 3);
    assert_eq!(vtx2rot.shape()[1], 3);
    assert_eq!(vtx2rot.shape()[2], 3);
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz_ini = vtx2xyz_ini.as_slice().unwrap();
    let vtx2xyz_def = vtx2xyz_def.as_slice().unwrap();
    let vtx2rot = vtx2rot.as_slice_mut().unwrap();
    del_fem_core::arap::optimal_rotations_mesh_vertx_for_arap_spoke_rim(
        vtx2rot,
        tri2vtx,
        vtx2xyz_ini,
        vtx2xyz_def,
    );
}

#[pyo3::pyfunction]
pub fn residual_arap_spoke_rim_trimesh3<'a>(
    _py: Python<'a>,
    tri2vtx: PyReadonlyArray2<'a, usize>,
    vtx2xyz_ini: PyReadonlyArray2<'a, f64>,
    vtx2xyz_def: PyReadonlyArray2<'a, f64>,
    vtx2rot: numpy::PyReadonlyArray3<'a, f64>,
    mut vtx2res: numpy::PyReadwriteArray2<'a, f64>,
) {
    assert!(tri2vtx.is_c_contiguous());
    assert_eq!(tri2vtx.shape().len(), 2);
    assert_eq!(tri2vtx.shape()[1], 3);
    assert!(vtx2xyz_ini.is_c_contiguous());
    assert!(vtx2xyz_def.is_c_contiguous());
    assert_eq!(vtx2xyz_ini.shape(), vtx2xyz_def.shape());
    assert_eq!(vtx2xyz_ini.shape().len(), 2);
    assert_eq!(vtx2xyz_ini.shape()[1], 3);
    assert!(vtx2rot.is_c_contiguous());
    assert_eq!(vtx2rot.shape().len(), 3);
    assert_eq!(vtx2rot.shape()[1], 3);
    assert_eq!(vtx2rot.shape()[2], 3);
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz_ini = vtx2xyz_ini.as_slice().unwrap();
    let vtx2xyz_def = vtx2xyz_def.as_slice().unwrap();
    let vtx2rot = vtx2rot.as_slice().unwrap();
    let vtx2res = vtx2res.as_slice_mut().unwrap();
    del_fem_core::arap::residual_arap_spoke_rim(
        vtx2res,
        tri2vtx,
        vtx2xyz_ini,
        vtx2xyz_def,
        vtx2rot,
    );
}
