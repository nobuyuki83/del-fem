import numpy
import scipy
import sys
#
import del_ls
from del_msh import TriMesh, PolyLoop
from del_fem import LaplacianMatrix, MassMatrix

def test_01():
    vtxi2xyi = numpy.array([
        [0, 0],
        [1, 0],
        [1, 0.6],
        [0.6, 0.6],
        [0.6, 1.0],
        [0, 1]], dtype=numpy.float32)
    ##
    tri2vtx, vtx2xy = PolyLoop.tesselation2d(
        vtxi2xyi, resolution_edge=0.04, resolution_face=0.04)
    print("# vtx: ", vtx2xy.shape[0])
    mmat = MassMatrix.from_uniform_mesh(tri2vtx, vtx2xy)
    smat = LaplacianMatrix.from_uniform_mesh(tri2vtx,vtx2xy)
    print(sys.path)
    assert scipy.sparse.linalg.norm(smat - smat.transpose()) < 1.0e-10
    assert scipy.linalg.norm(smat * numpy.ones((vtx2xy.shape[0]))) < 1.0e-4
    eig = scipy.sparse.linalg.eigsh(smat, M=mmat, sigma=0.0)
    i_eig = 4
    evec0 = eig[1].transpose()[i_eig].copy()
    eval0 = eig[0][i_eig]
    print(eval0)
    evec0 /= scipy.linalg.norm(evec0)
    assert abs(eval0*(mmat * evec0).dot(evec0)-(smat * evec0).dot(evec0)) < 1.0e-4