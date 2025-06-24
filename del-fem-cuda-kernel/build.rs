fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/laplacian_smoothing_jacobi.cu");
    println!("cargo:rerun-if-changed=src/pbd_rod2.cu");

    let path_out_dir = std::env::var("OUT_DIR").unwrap();
    //let path_out_dir = std::path::Path::new(&path_out_dir).join("cpp_headers").join("del_geo");
    let path_out_dir = std::path::Path::new(&path_out_dir).join("del_geo");
    // dbg!(&path_out_dir);
    std::fs::create_dir_all(&path_out_dir).unwrap();
    del_geo_cpp_headers::HEADERS.write_files(&path_out_dir);
    // dbg!("hoge");
    let glob_input = path_out_dir
        .join("*.h")
        .into_os_string()
        .into_string()
        .unwrap();
    let builder = bindgen_cuda::Builder::default().include_paths_glob(&glob_input);
    println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();
}
