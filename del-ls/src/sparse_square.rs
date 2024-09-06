//! sparse matrix class and functions

/// sparse matrix class
/// Compressed Row Storage (CRS) data structure
/// * `num_blk` - number of row and col blocks
pub struct Matrix<MAT> {
    pub num_blk: usize,
    pub row2idx: Vec<usize>,
    pub idx2col: Vec<usize>,
    pub idx2val: Vec<MAT>,
    pub row2val: Vec<MAT>,
}

impl<MAT> Matrix<MAT>
where
    MAT: 'static
        + num_traits::Zero
        + std::default::Default
        + std::ops::AddAssign // merge
        + Copy
        + std::fmt::Display,
{
    pub fn new() -> Self {
        Matrix {
            num_blk: 0,
            row2idx: vec![0],
            idx2col: Vec::<usize>::new(),
            idx2val: Vec::<MAT>::new(),
            row2val: Vec::<MAT>::new(),
        }
    }

    pub fn clone(&self) -> Self {
        Matrix {
            num_blk: self.num_blk,
            row2idx: self.row2idx.clone(),
            idx2col: self.idx2col.clone(),
            idx2val: self.idx2val.clone(),
            row2val: self.row2val.clone(),
        }
    }

    /// set non-zero pattern
    pub fn symbolic_initialization(&mut self, row2idx: &Vec<usize>, idx2col: &Vec<usize>) {
        self.num_blk = row2idx.len() - 1;
        self.row2idx = row2idx.clone();
        self.idx2col = idx2col.clone();
        let num_idx = self.row2idx[self.num_blk];
        assert_eq!(num_idx, idx2col.len());
        self.idx2val.resize_with(num_idx, Default::default);
        self.row2val.resize_with(self.num_blk, Default::default);
    }

    /// set zero to all the values
    pub fn set_zero(&mut self) {
        assert_eq!(self.idx2val.len(), self.idx2col.len());
        for m in self.row2val.iter_mut() {
            m.set_zero()
        }
        for m in self.idx2val.iter_mut() {
            m.set_zero()
        }
    }

    /// merge element-wise matrix to sparse matrix
    pub fn merge(
        &mut self,
        node2row: &[usize],
        node2col: &[usize],
        emat: &[MAT],
        merge_buffer: &mut Vec<usize>,
    ) {
        assert_eq!(emat.len(), node2row.len() * node2col.len());
        merge_buffer.resize(self.num_blk, usize::MAX);
        let col2idx = merge_buffer;
        for inode in 0..node2row.len() {
            let i_row = node2row[inode];
            assert!(i_row < self.num_blk);
            for ij_idx in self.row2idx[i_row]..self.row2idx[i_row + 1] {
                assert!(ij_idx < self.idx2col.len());
                let j_col = self.idx2col[ij_idx];
                col2idx[j_col] = ij_idx;
            }
            for jnode in 0..node2col.len() {
                let j_col = node2col[jnode];
                assert!(j_col < self.num_blk);
                if i_row == j_col {
                    // Marge Diagonal
                    self.row2val[i_row] += emat[inode * node2col.len() + jnode];
                } else {
                    // Marge Non-Diagonal
                    assert!(col2idx[j_col] < self.idx2col.len());
                    let ij_idx = col2idx[j_col];
                    assert_eq!(self.idx2col[ij_idx], j_col);
                    self.idx2val[ij_idx] += emat[inode * node2col.len() + jnode];
                }
            }
            for ij_idx in self.row2idx[i_row]..self.row2idx[i_row + 1] {
                assert!(ij_idx < self.idx2col.len());
                let j_col = self.idx2col[ij_idx];
                col2idx[j_col] = usize::MAX;
            }
        }
    }
}

/// generalized matrix-vector multiplication
/// where matrix is sparse (not block) matrix
/// `{y_vec} <- \alpha * [a_mat] * {x_vec} + \beta * {y_vec}`
pub fn mult_vec<T>(y_vec: &mut Vec<T>, beta: T, alpha: T, a_mat: &Matrix<T>, x_vec: &Vec<T>)
where
    T: std::ops::MulAssign // *=
        + std::ops::Mul<Output = T> // *
        + std::ops::AddAssign // +=
        + 'static
        + Copy, // =
    f32: num_traits::AsPrimitive<T>,
{
    assert_eq!(y_vec.len(), a_mat.num_blk);
    for m in y_vec.iter_mut() {
        *m *= beta;
    }
    for iblk in 0..a_mat.num_blk {
        for icrs in a_mat.row2idx[iblk]..a_mat.row2idx[iblk + 1] {
            assert!(icrs < a_mat.idx2col.len());
            let jblk0 = a_mat.idx2col[icrs];
            assert!(jblk0 < a_mat.num_blk);
            y_vec[iblk] += alpha * a_mat.idx2val[icrs] * x_vec[jblk0];
        }
        y_vec[iblk] += alpha * a_mat.row2val[iblk] * x_vec[iblk];
    }
}

pub fn mult_mat<T>(y_mat: &mut [T], beta: T, alpha: T, a_mat: &Matrix<T>, x_mat: &[T])
where
    T: std::ops::MulAssign // *=
        + std::ops::Mul<Output = T> // *
        + std::ops::AddAssign // +=
        + 'static
        + Copy, // =
    f32: num_traits::AsPrimitive<T>,
{
    let num_row = a_mat.row2idx.len() - 1;
    assert_eq!(y_mat.len(), x_mat.len());
    let num_dim = y_mat.len() / num_row;
    assert_eq!(y_mat.len(), num_dim * num_row);
    for val_y in y_mat.iter_mut() {
        *val_y *= beta;
    }
    for i_row in 0..num_row {
        for idx in a_mat.row2idx[i_row]..a_mat.row2idx[i_row + 1] {
            let j_col = a_mat.idx2col[idx];
            for y in 0..num_dim {
                y_mat[i_row * num_dim + y] +=
                    alpha * a_mat.idx2val[idx] * x_mat[j_col * num_dim + y];
            }
        }
        for y in 0..num_dim {
            y_mat[i_row * num_dim + y] += alpha * a_mat.row2val[i_row] * x_mat[i_row * num_dim + y];
        }
    }
}

#[test]
fn test_scalar() {
    let mut sparse = crate::sparse_square::Matrix::<f32>::new();
    let colind = vec![0, 2, 5, 8, 10];
    let rowptr = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    sparse.symbolic_initialization(&colind, &rowptr);
    sparse.set_zero();
    {
        let emat = [1., 0., 0., 1.];
        let mut tmp_buffer = Vec::<usize>::new();
        sparse.merge(&[0, 1], &[0, 1], &emat, &mut tmp_buffer);
    }
    let nblk = colind.len() - 1;
    let mut rhs = Vec::<f32>::new();
    rhs.resize(nblk, Default::default());
    let mut lhs = Vec::<f32>::new();
    lhs.resize(nblk, Default::default());
    mult_vec(&mut lhs, 1.0, 1.0, &sparse, &rhs);
}
