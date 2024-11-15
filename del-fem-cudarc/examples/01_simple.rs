use cudarc::driver::{CudaSlice, DeviceSlice};

fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz)
        = del_msh_core::trimesh3_primitive::sphere_yup::<u32, f32>(1., 32, 32);
    let num_vtx = vtx2xyz.len()/3;
    let (vtx2idx, idx2vtx)
        = del_msh_core::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx, false);
    let vtx2rhs = {
        let mut vtx2rhs = vec!(0f32; num_vtx);
        for i_vtx in 0..num_vtx {
            if vtx2xyz[i_vtx*3] > 0.8 {
                vtx2rhs[i_vtx] = 1.0;
            }
        }
        vtx2rhs
    };
    let lambda = 3f32;
    // ------------------
    let dev = cudarc::driver::CudaDevice::new(0)?;
    let vtx2idx_dev = dev.htod_copy(vtx2idx.clone())?;
    let idx2vtx_dev = dev.htod_copy(idx2vtx.clone())?;
    let vtx2rhs_dev = dev.htod_copy(vtx2rhs.clone())?;
    let mut vtx2lhs_ini: CudaSlice<f32> = dev.alloc_zeros(vtx2rhs_dev.len())?;
    let mut vtx2lhs_upd: CudaSlice<f32> = dev.alloc_zeros(vtx2lhs_ini.len())?;
    let mut vtx2res: CudaSlice<f32> = dev.alloc_zeros(vtx2lhs_ini.len())?;
    del_fem_cudarc::diffuse_jacobi::solve(
        &dev, &vtx2idx_dev, &idx2vtx_dev, lambda, &vtx2rhs_dev,
        &mut vtx2lhs_ini, &mut vtx2lhs_upd, &mut vtx2res)?;
    Ok(())
}