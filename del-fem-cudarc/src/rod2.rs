use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, PushKernelArg};
use del_cudarc_safe::cudarc;

pub fn solve(
    stream: &std::sync::Arc<CudaStream>,
    pnt2xy_ini: &CudaSlice<f32>,
    pnt2massinv: &CudaSlice<f32>,
    num_example: usize,
    dt: f32,
    gravity: &[f32; 2],
    example2pnt2xydef: &mut CudaViewMut<f32>,
    example2pnt2xynew: &mut CudaViewMut<f32>,
    example2pnt2velo: &mut CudaViewMut<f32>,
) -> Result<(), cudarc::driver::result::DriverError> {
    let num_point = pnt2xy_ini.len() / 2;
    dbg!(pnt2xy_ini.len());
    assert_eq!(pnt2massinv.len(), num_point);
    assert_eq!(example2pnt2xydef.len(), num_point * 2 * num_example);
    assert_eq!(example2pnt2xynew.len(), num_point * 2 * num_example);
    assert_eq!(example2pnt2xynew.len(), num_point * 2 * num_example);
    assert_eq!(example2pnt2velo.len(), num_point * 2 * num_example);
    let gravity = stream.memcpy_stod(gravity)?;
    let cfg = cudarc::driver::LaunchConfig::for_num_elems(num_example as u32);
    let func = del_cudarc_safe::get_or_load_func(
        &stream.context(),
        "solve",
        del_fem_cuda_kernel::PBD_ROD2,
    )?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&num_example);
    builder.arg(&num_point);
    builder.arg(pnt2xy_ini);
    builder.arg(pnt2massinv);
    builder.arg(&dt);
    builder.arg(&gravity);
    builder.arg(example2pnt2xydef);
    builder.arg(example2pnt2xynew);
    builder.arg(example2pnt2velo);
    unsafe { builder.launch(cfg)? };

    /*
    void solve(
        const uint32_t num_example,
        const uint32_t num_point,
        const float *pnt2xy_ini,
        const float *pnt2massinv,
        float dt,
        float *gravity,
        float *example2pnt2xydef,
        float *example2pnt2xynew,
        float *example2pnt2velo);
     */

    Ok(())
}
