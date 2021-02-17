#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};

mod clfind;

#[pymodule]
fn libclfind(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "clfind")]
    #[text_signature = "(hits)"]
    fn py_clfind(
        py: Python, hits: &PyArray2<i64>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        let labels: ndarray::Array1<i64> = clfind::clfind(hits.readonly().as_array());
        Ok(labels.to_pyarray(py).to_owned())
    }

    Ok(())
}
