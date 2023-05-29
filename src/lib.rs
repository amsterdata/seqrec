use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub mod types;
pub mod sparse_topk_index;
pub mod similar_row;
pub mod topk;
pub mod row_accumulator;
pub mod utils;
pub mod similarity;
pub mod serialize;
pub mod algorithms;

//use rayon::prelude::*;

use numpy::PyArrayDyn;
use sprs::CsMat;
use crate::sparse_topk_index::SparseTopKIndex;

#[pyfunction]
fn tifu_reps(
    basket_items_dict: &PyDict,
    user_baskets_dict: &PyDict,
    num_items: usize,
    m: usize,
    rb: f64,
    rg: f64,
) -> PyResult<Vec<Vec<f64>>> {

    eprintln!("{}", basket_items_dict.len());
    eprintln!("{}", user_baskets_dict.len());

    let num_users = user_baskets_dict.len();

    let user_reps: Vec<_> = (0..num_users)//TODO parallelize later .into_par_iter()
        .map(|user| {
            //let baskets: Vec<usize> = user_baskets_dict.get_item(user).unwrap().downcast::<PyList>().expect("This should not be here").extract().unwrap();
            let baskets: Vec<usize> = user_baskets_dict.get_item(user).unwrap().downcast::<PyList>().expect("This should not be here").extract().unwrap();
            algorithms::tifuknn::user_embedding(&baskets, &basket_items_dict, num_items, m, rb, rg)
        })
        .collect();

    Ok(user_reps)
}


#[pyclass]
struct Index {
    similarity_index: SparseTopKIndex,
}

#[pymethods]
impl Index {

    fn topk(&self, row: usize) -> PyResult<Vec<(usize,f32)>> {
        let topk: Vec<(usize,f32)> = self.similarity_index.neighbors(row)
            .map(|similar_user| (similar_user.row as usize, similar_user.similarity))
            .collect();
        Ok(topk)
    }

    fn forget(&mut self, row: usize, column: usize) -> PyResult<()> {
        self.similarity_index.forget(row, column);
        Ok(())
    }

    #[new]
    fn new(
        num_rows: usize,
        num_cols: usize,
        indptr: &PyArrayDyn<i32>,
        indices: &PyArrayDyn<i32>,
        data: &PyArrayDyn<f64>,
        k: usize
    ) -> Self {

        // TODO this horribly inefficient for now...
        let indices_copy: Vec<usize> = indices.to_vec().unwrap()
            .into_iter().map(|x| x as usize).collect();
        let indptr_copy: Vec<usize> = indptr.to_vec().unwrap()
            .into_iter().map(|x| x as usize).collect();
        let data_copy: Vec<f64> = data.to_vec().unwrap();

        let representations =
            CsMat::new((num_rows, num_cols), indptr_copy, indices_copy, data_copy);

        let similarity_index: SparseTopKIndex = SparseTopKIndex::new(representations, k);

        Self { similarity_index }
    }
}



#[pymodule]
fn seqrec(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tifu_reps, m)?)?;
    m.add_class::<Index>()?;
    Ok(())
}