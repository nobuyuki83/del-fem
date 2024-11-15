use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};

fn iter(
    dev: &std::sync::Arc<CudaDevice>,
    vtx2idx: &CudaSlice<u32>,
    idx2vtx: &CudaSlice<u32>,
    lambda: f32,
    vtx2rhs: &CudaSlice<f32>,
    vtx2lhs0: &mut CudaSlice<f32>,
    vtx2lhs1: &mut CudaSlice<f32>,
    vtx2res: &mut CudaSlice<f32>) -> anyhow::Result<()>
{
    let num_vtx = vtx2idx.len();
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_vtx as u32);
    let param = (
        num_vtx,
        vtx2idx,
        idx2vtx,
        lambda,
        vtx2rhs,
        vtx2lhs0,
        vtx2lhs1,
        vtx2res,
    );
    use cudarc::driver::LaunchAsync;
    let gpu_solve =
        crate::get_or_load_func(&dev, "solve_diffuse_jacobi", del_fem_cudarc_kernel::SIMPLE)?;
    unsafe { gpu_solve.launch(cfg, param) }?;
    Ok(())
}

pub fn solve(
    dev: &std::sync::Arc<CudaDevice>,
    vtx2idx: &CudaSlice<u32>,
    idx2vtx: &CudaSlice<u32>,
    lambda: f32,
    vtx2rhs: &CudaSlice<f32>,
    vtx2lhs0: &mut CudaSlice<f32>,
    vtx2lhs1: &mut CudaSlice<f32>,
    vtx2res: &mut CudaSlice<f32>) -> anyhow::Result<()>
{
    for i in 0..1000 {
        iter(dev, vtx2idx, idx2vtx, lambda, vtx2rhs, vtx2lhs0, vtx2lhs1, vtx2res)?;
        dev.dtod_copy(vtx2lhs1, vtx2lhs0)?;
        {
            let res: f32 = dev.dtoh_sync_copy(vtx2res)?.iter().map(|v| v*v).sum();
            dbg!(res);
        }
    }
    Ok(())
}