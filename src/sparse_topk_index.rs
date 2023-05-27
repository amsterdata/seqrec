use crate::similar_row::SimilarRow;
use crate::row_accumulator::RowAccumulator;
use crate::topk::{TopK, TopkUpdate};

use std::clone::Clone;
use std::collections::binary_heap::Iter;
use std::collections::BinaryHeap;
use sprs::CsMat;
use std::time::Instant;
use std::sync::Mutex;
use indicatif::{ProgressBar, ProgressStyle};

use std::env;
use core_affinity;

use num_cpus::get_physical;
use rayon::slice::Chunks;
use rayon::prelude::*;
use crate::similarity::{COSINE, Similarity};
use crate::topk::TopkUpdate::{NeedsFullRecomputation, NoChange, Update};
use crate::types::RowIndex;

use crate::utils::zero_out_entry;

pub struct SparseTopKIndex {
    pub representations: CsMat<f64>, // TODO required for one bench, fix this
    pub(crate) representations_transposed: CsMat<f64>,
    pub(crate) topk_per_row: Vec<TopK>,
    pub(crate) k: usize,
    pub(crate) norms: Vec<f64>,
}




impl SparseTopKIndex {

    pub fn neighbors(&self, row: usize) -> Iter<SimilarRow> {
        self.topk_per_row[row].iter()
    }

    fn configure_rayon(verbose: bool) {
        let num_threads = env::var("CABOOSE_NUM_THREADS")
            .map(|v: String| v.parse::<usize>().unwrap())
            .unwrap_or(num_cpus::get());

        let pin_threads = env::var("CABOOSE_PIN_THREADS")
            .map(|v: String| v.parse::<bool>().unwrap())
            .unwrap_or(false);

        if verbose {
            eprintln!("--Configuring for top-k -- num_threads: {}; pinning? {};",
                      num_threads, pin_threads);
        }

        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .start_handler(move |id| {
                //println!("Thread {} starting in pool", id);
                if pin_threads {
                    let core_ids = core_affinity::get_core_ids().unwrap();
                    //println!("Pinning thread {} to core {:?}", id, core_ids[id]);
                    core_affinity::set_for_current(core_ids[id]);
                }
            })
            .build()
            .unwrap();
    }

    fn parallel_topk<S: Similarity + Sync>(
        row_ranges: Chunks<usize>,
        representations: &CsMat<f64>,
        representations_transposed: &CsMat<f64>,
        num_rows: usize,
        k: usize,
        norms: &Vec<f64>,
        similarity: &S,
        topk_per_row: &mut Vec<TopK>,
    ) {
        let data = representations.data();
        let indices = representations.indices();
        let data_t = representations_transposed.data();
        let indices_t = representations_transposed.indices();
        let indptr_sprs = representations.indptr();
        let indptr_t_sprs = representations_transposed.indptr();

        let bar = ProgressBar::new(num_rows as u64);
        let template = "{wide_bar} | {pos}/{len} | Elapsed: {elapsed_precise}, ETA: {eta_precise}";
        bar.set_style(ProgressStyle::default_bar().template(template).unwrap());
        let shared_progress = Mutex::new(bar);

        // TODO use a thread local here for the accumulator
        let topk_partitioned: Vec<_> = row_ranges.map(|range| {

            let indptr = indptr_sprs.raw_storage();
            let indptr_t = indptr_t_sprs.raw_storage();

            // We need to get rid of these allocations and do them only once per thread
            let mut topk_per_row: Vec<TopK> = Vec::with_capacity(range.len());
            let mut accumulator = RowAccumulator::new(num_rows.clone());

            for row in range {

                let ptr_start = unsafe { *indptr.get_unchecked(*row) };
                let ptr_end = unsafe { *indptr.get_unchecked(*row + 1) };

                for ptr in ptr_start..ptr_end {
                    let value = unsafe { * data.get_unchecked(ptr) };

                    let other_ptr_start = unsafe { *indptr_t.get_unchecked(*indices.get_unchecked(ptr)) };
                    let other_ptr_end = unsafe { *indptr_t.get_unchecked(*indices.get_unchecked(ptr) + 1) };

                    for other_ptr in other_ptr_start..other_ptr_end {
                        let index = unsafe { *indices_t.get_unchecked(other_ptr) };
                        let value_t = unsafe { *data_t.get_unchecked(other_ptr) };
                        accumulator.add_to(index, value_t * value);
                    }
                }

                let topk = accumulator.topk_and_clear(*row, k, similarity, &norms);
                topk_per_row.push(topk);
                shared_progress.lock().unwrap().inc(1 as u64);
            }
            (range, topk_per_row)
        }).collect();

        for (range, topk_partition) in topk_partitioned.into_iter() {
            for (index, topk) in range.into_iter().zip(topk_partition.into_iter()) {
                topk_per_row[*index] = topk;
            }
        }
    }

    pub fn new(representations: CsMat<f64>, k: usize) -> Self {
        let (num_rows, _) = representations.shape();

        eprintln!("--Creating transpose of R...");
        let mut representations_transposed: CsMat<f64> = representations.to_owned();
        representations_transposed.transpose_mut();
        representations_transposed = representations_transposed.to_csr();

        // TODO Make configurable at some point
        let similarity = COSINE;

        eprintln!("--Computing row norms...");
        //TODO is it worth to parallelize this?
        let norms: Vec<f64> = (0..num_rows)
            .map(|row| {
                let mut norm_accumulator: f64 = 0.0;
                for column_index in representations.indptr().outer_inds_sz(row) {
                    let value = representations.data()[column_index];
                    norm_accumulator += similarity.accumulate_norm(value);
                }
                similarity.finalize_norm(norm_accumulator)
            })
            .collect();

        SparseTopKIndex::configure_rayon(true);

        let chunk_size = env::var("CABOOSE_TOPK_CHUNK_SIZE")
            .map(|v: String| v.parse::<usize>().unwrap())
            .unwrap_or(1024);

        let row_range = (0..num_rows).collect::<Vec<usize>>();
        let row_ranges = row_range.par_chunks(chunk_size);
        let mut topk_per_row: Vec<TopK> = vec![TopK::new(BinaryHeap::new()); num_rows];

        eprintln!("--Scheduling parallel top-k computation...");
        SparseTopKIndex::parallel_topk(
            row_ranges,
            &representations,
            &representations_transposed,
            num_rows.clone(),
            k,
            &norms,
            &similarity,
            &mut topk_per_row
        );

        Self {
            representations,
            representations_transposed,
            topk_per_row,
            k,
            norms,
        }
    }


    pub fn forget(&mut self, row: usize, column: usize) {

        SparseTopKIndex::configure_rayon(false);

        let similarity = COSINE;
        let (num_rows, _) = self.representations.shape();
        let old_value = self.representations.get(row, column);

        assert!(
            old_value.is_some(),
            "Cannot forget non-existing value of row {} and column {}!",
            row,
            column);

        let old_value = old_value.unwrap().clone();

        zero_out_entry(&mut self.representations, row, column);
        zero_out_entry(&mut self.representations_transposed, column, row);

        self.norms[row] = similarity.update_norm(self.norms[row], old_value);

        let data = self.representations.data();
        let indices = self.representations.indices();
        let indptr_sprs = self.representations.indptr();
        let data_t = self.representations_transposed.data();
        let indices_t = self.representations_transposed.indices();
        let indptr_t_sprs = self.representations_transposed.indptr();

        let start_time = Instant::now();

        let ptr_indices: Vec<_> = indptr_sprs.outer_inds_sz(row).collect();
        let num_cores = get_physical();
        let chunk_size = std::cmp::max(1, ptr_indices.len() / num_cores);

        let accs: Vec<_> = ptr_indices.par_chunks(chunk_size).map(|ptr_range| {

            let indptr_t = indptr_t_sprs.raw_storage();

            let mut accumulator = RowAccumulator::new(num_rows.clone());

            for ptr in ptr_range {
                let value = unsafe { *data.get_unchecked(*ptr) };
                let other_ptr_start = unsafe { *indptr_t.get_unchecked(*indices.get_unchecked(*ptr)) };
                let other_ptr_end = unsafe { *indptr_t.get_unchecked(*indices.get_unchecked(*ptr) + 1) };

                for other_row in other_ptr_start..other_ptr_end {
                    let index = unsafe { *indices_t.get_unchecked(other_row) };
                    let value_t = unsafe { *data_t.get_unchecked(other_row) };
                    accumulator.add_to(index, value_t * value);
                }
            }

            accumulator
        }).collect();

        let all_directly_affected_rows: Vec<usize> = indptr_t_sprs.outer_inds_sz(column)
            .map(|i| indices_t[i])
            .collect();

        let (updated_similarities, topk) = RowAccumulator::merge_and_collect_all(
            row,
            &similarity,
            self.k,
            &self.norms,
            accs,
            all_directly_affected_rows);

        let parallel_similarity_duration = (Instant::now() - start_time).as_millis();

        let start_time = Instant::now();
        let mut rows_to_fully_recompute = Vec::new();

        let changes: Vec<(RowIndex, TopkUpdate)> = updated_similarities.par_iter().map(|similar| {

            assert_ne!(similar.row, row as RowIndex);

            let other_row = similar.row;
            let similarity = similar.similarity;

            let other_topk = &self.topk_per_row[other_row as usize];
            let already_in_topk = other_topk.contains(row as RowIndex);

            let update = SimilarRow::new(row as RowIndex, similarity);

            let change = if !already_in_topk {
                if similarity != 0.0 {
                    assert_eq!(other_topk.len(), self.k,
                               "Invariant violated: row {} (with updated similarity {}) not in \
                               topk of row {}, but topk have length of {} instead of k={} only",
                               row, similarity, other_row, other_topk.len(), self.k);
                    other_topk.offer_non_existing_entry(update)
                } else {
                    NoChange
                }
            } else {
                if other_topk.len() < self.k {
                    if update.similarity == 0.0 {
                        other_topk.remove_existing_entry(update.row, self.k)
                    } else {
                        other_topk.update_existing_entry(update, self.k)
                    }
                } else {
                    if update.similarity == 0.0 {
                        NeedsFullRecomputation
                    } else {
                        other_topk.update_existing_entry(update, self.k)
                    }
                }
            };
            (other_row, change)
        }).collect();

        let change_duration = (Instant::now() - start_time).as_millis();

        let start_time = Instant::now();
        let mut count_nochange = 0;
        let mut count_update = 0;
        let mut count_recompute = 0;
        for (other_row, change) in changes {
            match change {
                NeedsFullRecomputation => {
                    rows_to_fully_recompute.push(other_row);
                    count_recompute += 1;
                },
                Update(new_topk) => {
                    count_update += 1;
                    self.topk_per_row[other_row as usize] = new_topk;
                },
                NoChange => {
                    count_nochange += 1;
                }
            }
        }

        self.topk_per_row[row] = topk;
        let change_apply_duration = (Instant::now() - start_time).as_millis();

        let indptr = indptr_sprs.raw_storage();
        let indptr_t = indptr_t_sprs.raw_storage();

        let mut accumulator = RowAccumulator::new(num_rows.clone());
        let start_time = Instant::now();
        // TODO is it worth to parallelize this?
        for row_to_recompute in rows_to_fully_recompute {

            let row_index = row_to_recompute as usize;

            let ptr_start = unsafe { *indptr.get_unchecked(row_index) };
            let ptr_end = unsafe { *indptr.get_unchecked(row_index + 1) };

            for ptr in ptr_start..ptr_end {
                let value = unsafe { data.get_unchecked(ptr) };

                let other_ptr_start = unsafe { *indptr_t.get_unchecked(*indices.get_unchecked(ptr)) };
                let other_ptr_end = unsafe { *indptr_t.get_unchecked(*indices.get_unchecked(ptr) + 1) };

                for other_ptr in other_ptr_start..other_ptr_end {
                    let index = unsafe { *indices_t.get_unchecked(other_ptr) };
                    let value_t = unsafe { *data_t.get_unchecked(other_ptr) };
                    accumulator.add_to(index, value_t * value);
                }
            }

            let topk = accumulator.topk_and_clear(row_index, self.k, &similarity, &self.norms);

            self.topk_per_row[row_index] = topk;
        }
        let recompute_duration = (Instant::now() - start_time).as_millis();

        let _changes = [count_nochange + count_update + count_recompute,
            count_nochange, count_update, count_recompute];
        let _durations = [parallel_similarity_duration, change_duration, change_apply_duration,
            recompute_duration];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;
    use sprs::TriMat;
    use crate::sparse_topk_index::SparseTopKIndex;
    use crate::types::Score;

    #[test]
    fn test_mini_example() {

        /*
        import numpy as np

        A = np.array(
                [[1, 1, 1, 0, 1],
                 [0, 1, 0, 1, 0],
                 [0, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        print(cosine)

        [[1.         0.35355339 0.8660254  0.        ]
         [0.35355339 1.         0.40824829 0.70710678]
         [0.8660254  0.40824829 1.         0.        ]
         [0.         0.70710678 0.         1.        ]]
        */

        let num_users = 4;
        let num_items = 5;

        let triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
            (1, 1, 1.0), (1, 3, 1.0),
            (2, 1, 1.0), (2, 2, 1.0), (2, 4, 1.0),
            (3, 3, 1.0),
        ];

        let mut input = TriMat::new((num_users, num_items));
        for (row, col, val) in triplets {
            input.add_triplet(row, col, val);
        }

        let user_representations = input.to_csr();
        let index = SparseTopKIndex::new(user_representations, 2);

        let mut n0: Vec<_> = index.neighbors(0).collect();
        n0.sort();
        assert_eq!(n0.len(), 2);
        check_entry(n0[0], 2, 0.8660254);
        check_entry(n0[1], 1, 0.35355339);

        let mut n1: Vec<_> = index.neighbors(1).collect();
        n1.sort();
        assert_eq!(n1.len(), 2);
        check_entry(n1[0], 3, FRAC_1_SQRT_2);
        check_entry(n1[1], 2, 0.40824829);

        let mut n2: Vec<_> = index.neighbors(2).collect();
        n2.sort();
        assert_eq!(n2.len(), 2);
        check_entry(n2[0], 0, 0.8660254);
        check_entry(n2[1], 1, 0.40824829);

        let n3: Vec<_> = index.neighbors(3).collect();
        assert_eq!(n3.len(), 1);
        check_entry(n3[0], 1, FRAC_1_SQRT_2);
    }

    #[test]
    fn test_mini_example_with_deletion() {

        /*
        import numpy as np

        A = np.array(
                [[1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0],
                 [0, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        print(cosine)

        [[1.         0.         0.66666667 0.        ]
         [0.         1.         0.40824829 0.70710678]
         [0.66666667 0.40824829 1.         0.        ]
         [0.         0.70710678 0.         1.        ]]
        */

        let num_users = 4;
        let num_items = 5;

        let triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
            (1, 1, 1.0), (1, 3, 1.0),
            (2, 1, 1.0), (2, 2, 1.0), (2, 4, 1.0),
            (3, 3, 1.0),
        ];

        let mut input = TriMat::new((num_users, num_items));
        for (row, col, val) in triplets {
            input.add_triplet(row, col, val);
        }

        let user_representations = input.to_csr();
        let mut index = SparseTopKIndex::new(user_representations, 2);

        index.forget(0, 1);

        let mut n0: Vec<_> = index.neighbors(0).collect();
        n0.sort();
        assert_eq!(n0.len(), 1);
        check_entry(n0[0], 2, 0.66666667);

        let mut n1: Vec<_> = index.neighbors(1).collect();
        n1.sort();
        assert_eq!(n1.len(), 2);
        check_entry(n1[0], 3, FRAC_1_SQRT_2);
        check_entry(n1[1], 2, 0.40824829);

        let mut n2: Vec<_> = index.neighbors(2).collect();
        n2.sort();
        assert_eq!(n2.len(), 2);
        check_entry(n2[0], 0, 0.66666667);
        check_entry(n2[1], 1, 0.40824829);

        let n3: Vec<_> = index.neighbors(3).collect();
        assert_eq!(n3.len(), 1);
        check_entry(n3[0], 1, FRAC_1_SQRT_2);
    }

    #[test]
    fn test_mini_example_with_double_deletion() {

        /*
        import numpy as np

        A = np.array(
                [[1, 0, 1, 0, 1],
                 [0, 1, 0, 0, 0],
                 [0, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        print(cosine)

        [[1.         0.         0.66666667 0.        ]
         [0.         1.         0.57735027 0.        ]
         [0.66666667 0.57735027 1.         0.        ]
         [0.         0.         0.         1.        ]]
        */

        let num_users = 4;
        let num_items = 5;

        let triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
            (1, 1, 1.0), (1, 3, 1.0),
            (2, 1, 1.0), (2, 2, 1.0), (2, 4, 1.0),
            (3, 3, 1.0),
        ];

        let mut input = TriMat::new((num_users, num_items));
        for (row, col, val) in triplets {
            input.add_triplet(row, col, val);
        }

        let user_representations = input.to_csr();
        let mut index = SparseTopKIndex::new(user_representations, 2);

        index.forget(0, 1);
        index.forget(1, 3);

        let mut n0: Vec<_> = index.neighbors(0).collect();
        n0.sort();
        assert_eq!(n0.len(), 1);
        check_entry(n0[0], 2, 0.66666667);

        let mut n1: Vec<_> = index.neighbors(1).collect();
        n1.sort();
        assert_eq!(n1.len(), 1);
        check_entry(n1[0], 2, 0.57735027);


        let mut n2: Vec<_> = index.neighbors(2).collect();
        n2.sort();
        assert_eq!(n2.len(), 2);
        check_entry(n2[0], 0, 0.66666667);
        check_entry(n2[1], 1, 0.57735027);

        let n3: Vec<_> = index.neighbors(3).collect();
        assert_eq!(n3.len(), 0);
        dbg!(n3);
    }


    fn check_entry(entry: &SimilarRow, expected_user: RowIndex, expected_similarity: Score) {
        assert_eq!(entry.row, expected_user);
        assert!((entry.similarity - expected_similarity).abs() < 0.0001);
    }
}

