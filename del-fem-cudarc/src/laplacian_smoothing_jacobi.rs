use cudarc::driver::{CudaDevice, CudaSlice, CudaViewMut, DeviceSlice};
use del_cudarc::cudarc;

pub fn solve(
    dev: &std::sync::Arc<CudaDevice>,
    vtx2idx: &CudaSlice<u32>,
    idx2vtx: &CudaSlice<u32>,
    lambda: f32,
    vtx2vars_next: &mut CudaViewMut<f32>,
    vtx2vars_prev: &CudaSlice<f32>,
    vtx2trgs: &CudaSlice<f32>,
) -> Result<(), cudarc::driver::result::DriverError> {
    let num_vtx: u32 = (vtx2idx.len() - 1) as u32;
    let num_dim: u32 = vtx2trgs.len() as u32 / num_vtx;
    assert_eq!((num_vtx * num_dim) as usize, vtx2vars_next.len());
    assert_eq!((num_vtx * num_dim) as usize, vtx2vars_prev.len());
    assert_eq!((num_vtx * num_dim) as usize, vtx2trgs.len());
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_vtx);
    let param = (
        num_vtx,
        vtx2idx,
        idx2vtx,
        lambda,
        vtx2vars_next,
        vtx2vars_prev,
        vtx2trgs,
    );
    use cudarc::driver::LaunchAsync;
    let gpu_solve = crate::get_or_load_func(
        &dev,
        "laplacian_smoothing_jacobi",
        del_fem_cudarc_kernel::LAPLACIAN_SMOOTHING_JACOBI,
    )?;
    unsafe { gpu_solve.launch(cfg, param) }?;
    Ok(())
}
