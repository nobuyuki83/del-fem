class Matrix {
    def __init__(self, row2idx, idx2col, block_size):
        self.num_blk = row2idx.shape[0] - 1;
        num_idx = idx2col.shape[0];
        self.row2idx = row2idx;
        self.idx2col = idx2col;
        self.row2val = numpy.ndarray(shape=(num_idx, block_size), dtype=numpy.float32, order='C');
        self.idx2val = numpy.ndarray(shape=(num_blk, block_size), dtype=numpy.float32, order='C');
        pass
}



