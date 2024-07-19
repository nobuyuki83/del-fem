import numpy
import scipy
import sys
#
import del_ls
from del_msh import TriMesh, PolyLoop

def test_01():
    tri2vtx, vtx2xyz = TriMesh.cylinder(radius=0.3, height=1.8, ndiv_height=32)
    vtx2idx, idx2vtx = TriMesh.vtx2vtx(tri2vtx, vtx2xyz.shape[0])
    sparse = del_ls.SparseSquareMatrix(vtx2idx, idx2vtx)
    sparse.set_zero()
    from del_fem.del_fem import merge_hessian_mesh_laplacian_on_trimesh3
    merge_hessian_mesh_laplacian_on_trimesh3(
        tri2vtx, vtx2xyz,
        sparse.row2idx, sparse.idx2col,
        sparse.row2val, sparse.idx2val)