macro_rules! get_cpu_slice_and_storage_from_tensor {
    ($slice: ident, $storage: ident, $tensor: expr, $t: ty) => {
        let $storage = $tensor.storage_and_layout().0;
        let $slice = match $storage.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<$t>()?,
            _ => panic!(),
        };
    };
}

#[cfg(feature = "cuda")]
macro_rules! get_cuda_slice_and_storage_and_layout_from_tensor {
    ($slice: ident, $storage: ident, $layout: ident, $tensor: expr, $t: ty) => {
        let ($storage, $layout) = $tensor.storage_and_layout();
        let $slice = match $storage.deref() {
            candle_core::Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<$t>()?,
            _ => panic!(),
        };
    };
}

#[cfg(feature = "cuda")]
macro_rules! get_cuda_slice_and_device_from_storage_u32 {
    ($slice: ident, $device: ident, $storage: expr) => {
        let CudaStorage { slice, device } = $storage;
        let ($slice, $device) = match slice {
            CudaStorageSlice::U32(slice) => (slice, device),
            _ => panic!(),
        };
    };
}

#[cfg(feature = "cuda")]
macro_rules! get_cuda_slice_and_device_from_storage_f32 {
    ($slice: ident, $device: ident, $storage: expr) => {
        let CudaStorage { slice, device } = $storage;
        let ($slice, $device) = match slice {
            CudaStorageSlice::F32(slice) => (slice, device),
            _ => panic!(),
        };
    };
}

mod laplacian_smoothing;
