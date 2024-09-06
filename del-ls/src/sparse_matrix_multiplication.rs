use num_traits::AsPrimitive;

/// non-zero pattern of spares matrix multiplication (`c = a * b`)
/// * `a_is_square` - matrix `a` has diagonal entry that is not listed in the CRS
/// * `b_is_square` - matrix `b` has diagonal entry that is not listed in the CRS
/// * `c_is_square` - if we exclude diagonal entry of matrix `c` from CRS
fn symbolic_multiplication(
    a_row2idx: &[usize],
    a_idx2col: &[usize],
    a_is_square: bool,
    b_row2idx: &[usize],
    b_idx2col: &[usize],
    b_is_square: bool,
    b_num_column: usize,
    c_is_square: bool,
) -> (Vec<usize>, Vec<usize>) {
    let nrow_a = a_row2idx.len() - 1;
    let mut c_row2idx = vec![0_usize; nrow_a + 1];
    let mut col2flag = vec![usize::MAX; b_num_column];
    for irow_a in 0..nrow_a {
        for &k_colrow in &a_idx2col[a_row2idx[irow_a]..a_row2idx[irow_a + 1]] {
            for &jcol_b in &b_idx2col[b_row2idx[k_colrow]..b_row2idx[k_colrow + 1]] {
                if col2flag[jcol_b] == irow_a || (jcol_b == irow_a && c_is_square) {
                    continue;
                }
                c_row2idx[irow_a + 1] += 1;
                col2flag[jcol_b] = irow_a;
            }
            if b_is_square {
                if col2flag[k_colrow] == irow_a || (k_colrow == irow_a && c_is_square) {
                    continue;
                }
                c_row2idx[irow_a + 1] += 1;
                col2flag[k_colrow] = irow_a;
            }
        }
        if a_is_square {
            for &jcol_b in &b_idx2col[b_row2idx[irow_a]..b_row2idx[irow_a + 1]] {
                if col2flag[jcol_b] == irow_a || (jcol_b == irow_a && c_is_square) {
                    continue;
                }
                c_row2idx[irow_a + 1] += 1;
                col2flag[jcol_b] = irow_a;
            }
            if b_is_square {
                if col2flag[irow_a] == irow_a || c_is_square {
                    continue;
                }
                c_row2idx[irow_a + 1] += 1;
                col2flag[irow_a] = irow_a;
            }
        }
    }
    // ---------
    for irow_a in 0..nrow_a {
        c_row2idx[irow_a + 1] += c_row2idx[irow_a];
    }
    let mut c_idx2col = vec![0_usize; c_row2idx[nrow_a]];
    // ---------
    col2flag.iter_mut().for_each(|v| *v = usize::MAX);
    for irow_a in 0..nrow_a {
        for &k in &a_idx2col[a_row2idx[irow_a]..a_row2idx[irow_a + 1]] {
            for &jcol_b in &b_idx2col[b_row2idx[k]..b_row2idx[k + 1]] {
                if col2flag[jcol_b] == irow_a || (jcol_b == irow_a && c_is_square) {
                    continue;
                }
                let c_ind = c_row2idx[irow_a];
                c_row2idx[irow_a] += 1;
                c_idx2col[c_ind] = jcol_b;
                col2flag[jcol_b] = irow_a;
            }
            if b_is_square {
                if col2flag[k] == irow_a || (k == irow_a && c_is_square) {
                    continue;
                }
                let c_ind = c_row2idx[irow_a];
                c_row2idx[irow_a] += 1;
                c_idx2col[c_ind] = k;
                col2flag[k] = irow_a;
            }
        }
        if a_is_square {
            for &jcol_b in &b_idx2col[b_row2idx[irow_a]..b_row2idx[irow_a + 1]] {
                if col2flag[jcol_b] == irow_a || (jcol_b == irow_a && c_is_square) {
                    continue;
                }
                let c_ind = c_row2idx[irow_a];
                c_row2idx[irow_a] += 1;
                c_idx2col[c_ind] = jcol_b;
                col2flag[jcol_b] = irow_a;
            }
            if b_is_square {
                if col2flag[irow_a] == irow_a || c_is_square {
                    continue;
                }
                let c_ind = c_row2idx[irow_a];
                c_row2idx[irow_a] += 1;
                c_idx2col[c_ind] = irow_a;
                col2flag[irow_a] = irow_a;
            }
        }
    }
    for irow0 in (1..nrow_a).rev() {
        c_row2idx[irow0] = c_row2idx[irow0 - 1];
    }
    c_row2idx[0] = 0;
    (c_row2idx, c_idx2col)
}

pub fn mult_square_matrices<T>(
    m0: &crate::sparse_square::Matrix<T>,
    m1: &crate::sparse_square::Matrix<T>,
) -> crate::sparse_square::Matrix<T>
where
    T: 'static
        + Copy
        + std::ops::Mul<Output = T>
        + std::ops::AddAssign
        + std::clone::Clone
        + num_traits::Zero,
    f32: AsPrimitive<T>,
{
    let num_blk = m0.num_blk;
    assert_eq!(num_blk, m1.num_blk);
    let (row2idx, idx2col) = symbolic_multiplication(
        &m0.row2idx,
        &m0.idx2col,
        true,
        &m1.row2idx,
        &m1.idx2col,
        true,
        m1.num_blk,
        true,
    );

    let mut idx2val = vec![T::zero(); idx2col.len()];
    let mut row2val = vec![T::zero(); num_blk];

    let mut col2idx = vec![usize::MAX; num_blk];
    for irow0 in 0..m0.num_blk {
        for idx0 in row2idx[irow0]..row2idx[irow0 + 1] {
            let icol0 = idx2col[idx0];
            col2idx[icol0] = idx0;
        }
        // ----
        for m0_idx in m0.row2idx[irow0]..m0.row2idx[irow0 + 1] {
            let k = m0.idx2col[m0_idx];
            for m1_idx in m1.row2idx[k]..m1.row2idx[k + 1] {
                let jcol0 = m1.idx2col[m1_idx];
                if irow0 == jcol0 {
                    row2val[irow0] += m0.idx2val[m0_idx] * m1.idx2val[m1_idx];
                } else {
                    let idx0 = col2idx[jcol0];
                    idx2val[idx0] += m0.idx2val[m0_idx] * m1.idx2val[m1_idx];
                }
            }
            if irow0 == k {
                row2val[irow0] += m0.idx2val[m0_idx] * m1.row2val[k];
            } else {
                let idx0 = col2idx[k];
                idx2val[idx0] += m0.idx2val[m0_idx] * m1.row2val[k];
            }
        }
        for m1_idx in m1.row2idx[irow0]..m1.row2idx[irow0 + 1] {
            let jcol0 = m1.idx2col[m1_idx];
            if irow0 == jcol0 {
                row2val[irow0] += m0.row2val[irow0] * m1.idx2val[m1_idx];
            } else {
                let idx0 = col2idx[jcol0];
                idx2val[idx0] += m0.row2val[irow0] * m1.idx2val[m1_idx];
            }
        }
        row2val[irow0] += m0.row2val[irow0] * m1.row2val[irow0];
        // ----
        for idx0 in row2idx[irow0]..row2idx[irow0 + 1] {
            let icol0 = idx2col[idx0];
            col2idx[icol0] = usize::MAX;
        }
    }

    crate::sparse_square::Matrix::<T> {
        num_blk,
        row2idx,
        idx2col,
        idx2val,
        row2val,
    }
}
