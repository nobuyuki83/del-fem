use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyAnyMethods};
use dlpack::ManagedTensor;

#[cfg(feature = "cuda")]
use del_cudarc::cudarc;

#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;

/// Pythonから渡された PyCapsule を Rust 側で読み取る
#[pyfunction]
fn set_consecutive_sequence(_py: Python, obj: &pyo3::Bound<'_,PyAny>) -> PyResult<()> {
    let capsule = obj.downcast::<PyCapsule>()?;

    println!("Capsule name: {}", capsule.name()?.unwrap().to_str()?);

    // DLPack を unsafe にアンラップ
    unsafe {
        let ptr = capsule.pointer() as *mut ManagedTensor;
        if ptr.is_null() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Null ManagedTensor"));
        }
        let tensor = &(*ptr).dl_tensor;
        let ndim = tensor.ndim as usize;
        let shape = std::slice::from_raw_parts(tensor.shape, ndim);
        let total_elements = shape.iter().product::<i64>() as usize;
        println!("DLPack tensor shape: {:?}, {:?}, {:?}", ndim, shape, total_elements);
        let data_ptr = tensor.data as *mut f32;
        match tensor.ctx.device_type {
            dlpack::device_type_codes::CPU => {
                println!("CPU");
                let data = std::slice::from_raw_parts_mut(data_ptr, total_elements);
                println!("Read data from DLPack:s {:?}", data);

            }
            #[cfg(feature = "cuda")]
            dlpack::device_type_codes::GPU => {
                println!("GPU_{}", tensor.ctx.device_id);
                let ctx = cudarc::driver::CudaContext::new(tensor.ctx.device_id as usize).unwrap();
                let stream = ctx.default_stream();
                let mut data = del_cudarc::util::from_raw_parts_mut::<u32>(stream.clone(), data_ptr as u64, total_elements);
                del_cudarc::util::set_consecutive_sequence(&stream, &mut data).unwrap();
            },
            _ => println!("Unknown device type"),
        }
    }

    Ok(())
}

/// Pythonモジュールに登録
#[pymodule]
#[pyo3(name = "del_fem_dlpack")]
fn del_fem_dlpack_(_py: Python, m: &Bound<'_,PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_consecutive_sequence, m)?)?;
    Ok(())
}