#[cfg(feature = "cuda")]
use del_cudarc_safe::cudarc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DeviceSlice};

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh3_primitive::sphere_yup::<u32, f32>(1., 32, 32);
    let num_vtx = vtx2xyz.len() / 3;
    let (vtx2idx, idx2vtx) = del_msh_cpu::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx, false);
    let vtx2rhs = {
        let mut vtx2rhs = vec![0f32; num_vtx];
        for i_vtx in 0..num_vtx {
            if vtx2xyz[i_vtx * 3] > 0.8 {
                vtx2rhs[i_vtx] = 1.0;
            }
        }
        vtx2rhs
    };
    let lambda = 3f32;
    // ------------------
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let vtx2idx_dev = stream.memcpy_stod(&vtx2idx)?;
    let idx2vtx_dev = stream.memcpy_stod(&idx2vtx)?;
    let vtx2rhs_dev = stream.memcpy_stod(&vtx2rhs)?;
    let mut vtx2lhs0_dev: CudaSlice<f32> = stream.alloc_zeros(vtx2rhs_dev.len())?;
    let mut vtx2lhs1_dev: CudaSlice<f32> = stream.alloc_zeros(vtx2lhs0_dev.len())?;
    del_fem_cudarc::laplacian_smoothing_jacobi::solve(
        &stream,
        &vtx2idx_dev,
        &idx2vtx_dev,
        lambda,
        &mut vtx2lhs0_dev.slice_mut(..),
        &mut vtx2lhs1_dev,
        &vtx2rhs_dev,
    )?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {}
