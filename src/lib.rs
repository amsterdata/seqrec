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

    /*
    basket_reps = {}

    for basket in basket_items_dict:
        rep = np.zeros(self.num_items)
        for item in basket_items_dict[basket]:
            rep[item] = 1
        basket_reps[basket] = rep
    */
/*
    let mut basket_reps = HashMap::with_capacity(basket_items_dict.len());

    for el in basket_items_dict.items().iter() {
        let mut rep = vec![0; num_items];
        let tuple = el.downcast::<PyTuple>().unwrap();
        let basket = tuple.get_item(0).unwrap().extract::<usize>().unwrap();
        let basket_items: Vec<usize> = tuple.get_item(1).unwrap().downcast::<PyList>()?.extract().unwrap();

        for item in basket_items {
            rep[item] = 1;
        }

        basket_reps.insert(basket, rep);
    }
*/

    /*
    user_reps = []

    for user in range(self.num_users):

        rep = np.zeros(self.num_items)

        baskets = user_baskets_dict[user]
        group_size = math.ceil(len(baskets) / self.m)
        addition = (group_size * self.m) - len(baskets)

        basket_groups = []
        basket_groups.append(baskets[:group_size - addition])
        for i in range(self.m - 1):
            basket_groups.append(baskets[group_size - addition + (i * group_size):group_size - addition + ((i + 1) * group_size)])

        for i in range(self.m):
            group_rep = np.zeros(self.num_items)
            for j in range(1, len(basket_groups[i]) + 1):
                basket = basket_groups[i][j-1]
                basket_rep = np.array(basket_reps[basket]) * math.pow(self.rb, group_size-j)
                group_rep += basket_rep
            group_rep /= group_size

            rep += np.array(group_rep) * math.pow(self.rg, self.m-i)

        rep /= self.m
        user_reps.append(rep)
    */
    let num_users = user_baskets_dict.len();
    let mut user_reps = vec![vec![0.0;num_items]; num_users];

    for user in 0..num_users {

        /*
        baskets = user_baskets_dict[user]
        group_size = math.ceil(len(baskets) / self.m)
        addition = (group_size * self.m) - len(baskets)
        */
        //let py_baskets: &PyList = (user_baskets_dict.get_item(user).unwrap().downcast::<PyList>()?).clone();
        let baskets: Vec<usize> = user_baskets_dict.get_item(user).unwrap().downcast::<PyList>()?.extract().unwrap();

        let t = baskets.len();
        let group_size = ((t as f64) / m as f64).ceil() as usize;
        let num_groups = (t as f64 / group_size as f64).ceil() as usize;
        let size_of_first_group = t % group_size;

        let mut basket_groups = Vec::new();

        //eprintln!("#baskets {}, m {}, group size {}, size_of_first_group {}", baskets.len(), m, group_size, size_of_first_group);

        if size_of_first_group > 0 {
            let slice = baskets[0..size_of_first_group].to_vec();
            basket_groups.push(slice);
        }

        let skip = if size_of_first_group > 0 { 1 } else { 0 };

        for i in 0..(num_groups - skip) {
            let start = size_of_first_group + (i * group_size);
            let end = size_of_first_group + ((i + 1) * group_size);
            let slice = baskets[start..end].to_vec();
            basket_groups.push(slice);
        }

        /*
            for i in range(self.m):
                group_rep = np.zeros(self.num_items)
                for j in range(1, len(basket_groups[i]) + 1):
                    basket = basket_groups[i][j-1]
                    basket_rep = np.array(basket_reps[basket]) * math.pow(self.rb, group_size-j)
                    group_rep += basket_rep
                group_rep /= group_size

                rep += np.array(group_rep) * math.pow(self.rg, self.m-i)

            rep /= self.m
            user_reps.append(rep)
        */
        for i in 0..num_groups {
            let mut group_rep = vec![0.0; num_items];
            for j in 1..(basket_groups[i].len() + 1) {
                let basket: usize = basket_groups[i][j - 1].clone();
                let basket_items: Vec<usize> = basket_items_dict.get_item(basket).unwrap().downcast::<PyList>()?.extract().unwrap();
                for item in basket_items {
                    group_rep[item] += rb.powi((group_size - j) as i32);
                }
                /*for k in 0..num_items {
                    group_rep[k] += basket_reps.get(&basket).unwrap()[k] as f64 * rb.powi((group_size - j) as i32);
                }*/
            }
            for k in 0..num_items {
                group_rep[k] /= group_size as f64;
                user_reps[user][k] += group_rep[k] * rg.powi((m - i) as i32);
            }
        }
        for k in 0..num_items {
            user_reps[user][k] /= m as f64;
        }
    }

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