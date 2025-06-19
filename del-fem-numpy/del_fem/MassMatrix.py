import numpy

def from_uniform_mesh(elem2vtx, vtx2xyz, ndim=1):
    from del_msh.TriMesh import vtx2area
    vtx2area = vtx2area(elem2vtx, vtx2xyz)
    from scipy.sparse import dia_matrix
    mmat = dia_matrix(
        (vtx2area.repeat(ndim).reshape(1, -1), numpy.array([0])),
        shape=(vtx2xyz.shape[0]*ndim, vtx2xyz.shape[0]*ndim))
    return mmat