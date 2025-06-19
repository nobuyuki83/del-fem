use del_cudarc_safe::cudarc as cudarc;
use crate::cudarc::driver::CudaSlice;
fn main() -> anyhow::Result<()>{
    let gravity = [0., -10.];
    let dt = 0.01;
    let pnt2xy_ini = {
        let num_edge = 10;
        let num_pnt = num_edge + 1;
        let theta = std::f32::consts::PI / 6f32;
        let len_edge = 1f32 / (num_edge as f32);
        let mut pnt2xy = Vec::<f32>::with_capacity(num_pnt * 2);
        for i_pnt in 0..num_pnt {
            let x = 0.0f32 + len_edge * (i_pnt as f32) * theta.cos();
            let y = 0.8f32 - len_edge * (i_pnt as f32) * theta.sin();
            pnt2xy.push(x);
            pnt2xy.push(y);
        }
        pnt2xy
    };
    let pnt2massinv = {
        let mut pnt2massinv = vec![1f32; pnt2xy_ini.len() / 2];
        pnt2massinv[0] = 0f32;
        pnt2massinv
    };
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let num_example = 10;
    let pnt2xy_ini_gpu = stream.memcpy_stod(&pnt2massinv)?;
    let pnt2massinv_gpu = stream.memcpy_stod(&pnt2massinv)?;
    let mut pnt2xy_def: CudaSlice<f32> = unsafe { stream.alloc(pnt2xy_ini.len() * num_example)? };
    let mut pnt2xy_new: CudaSlice<f32> = unsafe { stream.alloc(pnt2xy_ini.len() * num_example)? };
    let mut pnt2velo: CudaSlice<f32> = stream.alloc_zeros(pnt2xy_ini.len() * num_example)?;
    Ok(())
}