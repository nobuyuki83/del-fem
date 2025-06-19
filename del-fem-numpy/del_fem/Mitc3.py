import numpy


def stiffness_matrix_from_uniform_mesh(
        thick,
        lam, myu,
        tri2vtx: numpy.typing.NDArray,
        vtx2xy: numpy.typing.NDArray):
    from del_msh.TriMesh import vtx2vtx
    row2idx, idx2col = vtx2vtx(tri2vtx, vtx2xy.shape[0], True)
    idx2val = numpy.zeros((idx2col.shape[0], 3, 3), dtype=numpy.float32)
    from del_fem.del_fem import merge_mitc3_to_bsr_for_meshtri2
    merge_mitc3_to_bsr_for_meshtri2(
        thick, lam, myu,
        tri2vtx, vtx2xy,
        row2idx, idx2col, idx2val)
    from scipy.sparse import bsr_matrix
    return bsr_matrix((idx2val, idx2col, row2idx))


def mass_matrix_from_uniform_mesh(
        thick,
        rho,
        elem2vtx: numpy.typing.NDArray,
        vtx2xyz: numpy.typing.NDArray):
    num_vtx = vtx2xyz.shape[0]
    vtx2mass = numpy.zeros((num_vtx * 3), dtype=numpy.float32)
    from del_fem.del_fem import mitc3_mass_for_trimesh2
    mitc3_mass_for_trimesh2(thick, rho, elem2vtx, vtx2xyz, vtx2mass)
    from scipy.sparse import dia_matrix
    mmat = dia_matrix(
        (vtx2mass.reshape(1, -1), numpy.array([0])),
        shape=(num_vtx * 3, num_vtx * 3))
    return mmat
