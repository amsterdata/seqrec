use pyo3::types::{PyDict, PyList};

pub fn user_embedding(
    baskets: &Vec<usize>,
    basket_items_dict: &PyDict,
    num_items: usize,
    m: usize,
    rb: f64,
    rg: f64,
) -> Vec<f64> {

    let mut embedding = vec![0.0; num_items];

    let t = baskets.len();
    let group_size = ((t as f64) / m as f64).ceil() as usize;
    let num_groups = (t as f64 / group_size as f64).ceil() as usize;
    let size_of_first_group = t % group_size;

    let mut basket_group_offsets = Vec::with_capacity(num_groups);

    if size_of_first_group > 0 {
        basket_group_offsets.push(size_of_first_group);
    }

    let skip = if size_of_first_group > 0 { 1 } else { 0 };

    for i in 0..(num_groups - skip) {
        let end = size_of_first_group + ((i + 1) * group_size);
        basket_group_offsets.push(end);
    }

    let mut i = 0;
    let mut start_index = 0;
    for offset in basket_group_offsets {
        let mut group_rep = vec![0.0; num_items];
        let basket_group = &baskets[start_index..offset];

        for j in 0..basket_group.len() {
            let basket = basket_group[j];
            let basket_items: Vec<usize> = basket_items_dict.get_item(basket).unwrap().downcast::<PyList>().expect("this should not be here").extract().unwrap();
            for item in basket_items {
                group_rep[item] += rb.powi((group_size - (j + 1)) as i32);
            }
        }
        for k in 0..num_items {
            group_rep[k] /= group_size as f64;
            embedding[k] += group_rep[k] * rg.powi((m - i) as i32);
        }

        start_index = offset;
        i += 1;
    }


    for k in 0..num_items {
        embedding[k] /= m as f64;
    }

    embedding
}