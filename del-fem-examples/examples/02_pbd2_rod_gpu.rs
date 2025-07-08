#[cfg(feature = "cuda")]
use crate::cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
use del_cudarc_safe::cudarc;

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let gravity = [0., -10.];
    let dt = 0.01;
    let num_point = 11;
    let pnt2xyini = {
        let num_edge = num_point - 1;
        let theta = std::f32::consts::PI / 6f32;
        let len_edge = 1f32 / (num_edge as f32);
        let mut pnt2xy = Vec::<f32>::with_capacity(num_point * 2);
        for i_pnt in 0..num_point {
            let x = 0.0f32 + len_edge * (i_pnt as f32) * theta.cos();
            let y = 0.8f32 - len_edge * (i_pnt as f32) * theta.sin();
            pnt2xy.push(x);
            pnt2xy.push(y);
        }
        pnt2xy
    };
    let pnt2massinv = {
        let mut pnt2massinv = vec![1f32; pnt2xyini.len() / 2];
        pnt2massinv[0] = 0f32;
        pnt2massinv
    };
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let num_example = 10000;
    let pnt2xyini_gpu = stream.memcpy_stod(&pnt2xyini)?;
    let pnt2massinv_gpu = stream.memcpy_stod(&pnt2massinv)?;
    let mut example2pnt2xydef_gpu: CudaSlice<f32> =
        unsafe { stream.alloc(pnt2xyini.len() * num_example)? };
    let mut example2pnt2xynew_gpu: CudaSlice<f32> =
        unsafe { stream.alloc(pnt2xyini.len() * num_example)? };
    let mut example2pnt2velo_gpu: CudaSlice<f32> =
        stream.alloc_zeros(pnt2xyini.len() * num_example)?;
    let now = std::time::Instant::now();
    del_fem_cudarc::rod2::solve(
        &stream,
        &pnt2xyini_gpu,
        &pnt2massinv_gpu,
        num_example,
        dt,
        &gravity,
        &mut example2pnt2xydef_gpu.as_view_mut(),
        &mut example2pnt2xynew_gpu.as_view_mut(),
        &mut example2pnt2velo_gpu.as_view_mut(),
    )?;
    dbg!(now.elapsed());
    /*
    let example2pnt2xydef = stream.memcpy_dtov(&example2pnt2xydef_gpu)?;
    for i_example in 0..num_example {
        for i_point in 0..num_point {
            println!("{} {} {}",
                i_point,
                example2pnt2xydef[(i_example*num_point+i_point)*2+0],
                     example2pnt2xydef[(i_example*num_point+i_point)*2+1],
            );
        }
    }
     */
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("this example need the cuda features as \"--features cuda\"");
}
