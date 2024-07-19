import numpy.typing
from scipy import sparse

def from_uniform_mesh(
        elem2vtx: numpy.typing.NDArray,
        vtx2xy: numpy.typing.NDArray):
    from del_msh import TriMesh
    row2idx, idx2col = TriMesh.vtx2vtx(elem2vtx, vtx2xy.shape[0], True)
    idx2val = numpy.zeros(idx2col.shape, dtype=numpy.float32)
    if elem2vtx.shape[1] == 3:
        if vtx2xy.shape[1] == 2:
            from del_fem.del_fem import merge_laplace_to_bsr_for_meshtri2
            merge_laplace_to_bsr_for_meshtri2(elem2vtx, vtx2xy, row2idx, idx2col, idx2val)
        elif vtx2xy.shape[1] == 3:
            from del_fem.del_fem import merge_laplace_to_bsr_for_meshtri3
            merge_laplace_to_bsr_for_meshtri3(elem2vtx, vtx2xy, row2idx, idx2col, idx2val)
    else:
        print("Error-> Not implemented")
    from scipy.sparse import csr_matrix
    smat = csr_matrix((idx2val, idx2col, row2idx))
    return smat