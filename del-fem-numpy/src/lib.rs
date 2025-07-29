mod arap;
mod block_sparse;
mod laplacian;
mod linear_solid;
mod mitc3;
mod rod3_darboux;

/// A Python module implemented in Rust.
#[pyo3::pymodule]
#[pyo3(name = "del_fem")]
fn del_fem_(_py: pyo3::Python, m: &pyo3::Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    laplacian::add_functions(_py, m)?;
    rod3_darboux::add_functions(_py, m)?;
    mitc3::add_functions(_py, m)?;
    linear_solid::add_functions(_py, m)?;
    block_sparse::add_functions(_py, m)?;
    arap::add_functions(_py, m)?;
    Ok(())
}
