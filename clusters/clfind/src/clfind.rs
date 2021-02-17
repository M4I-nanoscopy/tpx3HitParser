use ndarray::{arr1, arr2, Array1, ArrayView2, Axis};

pub fn clfind(hits: ArrayView2<i64>, time_window: i64) -> ndarray::Array1<i64> {
    let mut labels = Array1::from_elem(hits.len_of(Axis(0)), -1);
    let mut cluster_index = 0;
    let mut s: usize = 1;
    let mut i = 0;

    while i < hits.len_of(Axis(0)) {
        let hit = hits.index_axis(Axis(0), i);
        if labels[i] == -1 {
            // Unlabeled hit, start finding a cluster
            clfind_sub(i, hit, cluster_index, &mut s, &mut labels, hits, time_window);

            // Check if we can skip some hits that have now been marked as a cluster
            i = s.max(i);

            // Start marking the next cluster with new label
            cluster_index += 1;
        } else {
            i += 1;
        }
    }
    labels
}

fn clfind_sub(
    i: usize, h0: ndarray::ArrayView1<i64>, c_id: i64, start_idx: &mut usize,
    ls: &mut ndarray::Array1<i64>, hs: ndarray::ArrayView2<i64>, time_window: i64
) {
    let (x0, y0, t0) = (h0[0], h0[1], h0[2]);

    // Mark cluster in labels
    ls[i] = c_id;

    let mut j = *start_idx - 1;
    let mut first_new_start = true;

    // Loop over hits, looking for unlabelled hit that is within time window and a neighbor
    for i in *start_idx..hs.len_of(Axis(0)) {
        j = j + 1;

        if ls[j] != -1 {
            // Put the new start only at the first item not already labeled
            if first_new_start {
                *start_idx = j;
            }
            continue;
        } else {
            first_new_start = false;
        }

        let h1 = hs.index_axis(Axis(0), i);
        let (x1, y1, t1) = (h1[0], h1[1], h1[2]);

        // If we encounter a hit beyond our time window, we can stop searching. This assumes the hits are sorted on time
        if t1 > t0 + time_window {
            break;
        }

        // Look for neighbor, and if found search for its neighbors recursively
        if ((x0 - x1).abs() <= 1) && ((y0 - y1).abs() <= 1) {
            let tmp_start_idx = clfind_sub(j, h1, c_id, start_idx, ls, hs, time_window);
        }
    }
}
