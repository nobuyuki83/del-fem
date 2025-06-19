import numpy

def stiffness_matrix_from_uniform_mesh(
    tri2vtx: numpy.typing.NDArray,
    vtx2xy: numpy.typing.NDArray):
    from del_msh.TriMesh import vtx2vtx
    row2idx, idx2col = vtx2vtx(tri2vtx, vtx2xy.shape[0], True)
    idx2val = numpy.zeros((idx2col.shape[0],2,2), dtype=numpy.float32)
    from del_fem.del_fem import merge_linear_solid_to_csr_for_meshtri2
    merge_linear_solid_to_csr_for_meshtri2(tri2vtx, vtx2xy, row2idx, idx2col, idx2val)
    from scipy.sparse import bsr_matrix
    return bsr_matrix((idx2val, idx2col, row2idx))
