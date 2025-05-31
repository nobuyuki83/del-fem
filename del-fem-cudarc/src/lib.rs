#[cfg(feature = "cuda")]
pub mod laplacian_smoothing_jacobi;

#[cfg(feature = "cuda")]
use del_cudarc_safe::cudarc;


/*
#[cfg(feature = "cuda")]
pub fn get_or_load_func(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    module_name: &str,
    ptx: &'static str,
) -> Result<cudarc::driver::CudaFunction, cudarc::driver::result::DriverError> {
    if !dev.has_func(module_name, module_name) {
        // Leaking the string here is a bit sad but we need a &'static str and this is only
        // done once per kernel name.
        let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
        dev.load_ptx(ptx.into(), module_name, &[static_module_name])?
    }
    Ok(dev.get_func(module_name, module_name).unwrap())
}
 */

