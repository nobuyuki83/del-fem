import numpy

class SparseSquare:

    def __init__(self, row2idx, idx2col, ndof_block: int):
        num_vtx = row2idx.shape[0] - 1
        self.row2idx = row2idx
        self.idx2col = idx2col
        block_size = ndof_block * ndof_block
        self.row2val = numpy.ndarray(shape=(num_vtx,block_size), dtype=numpy.float32)
        num_idx = self.idx2col.shape[0]
        self.idx2val = numpy.ndarray(shape=(num_idx,block_size), dtype=numpy.float32)

    def set_zero(self):
        self.row2val.fill(0.)
        self.idx2val.fill(0.)