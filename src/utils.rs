use sprs::CsMat;

pub(crate) fn zero_out_entry(sparse_matrix: &mut CsMat<f64>, row: usize, column: usize) {
    // For some reason, the set method in sprs don't work...
    let index = sparse_matrix.nnz_index_outer_inner(row, column).unwrap();
    sparse_matrix.data_mut()[index.0] = 0.0;
}
