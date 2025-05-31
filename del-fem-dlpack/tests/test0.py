import torch
from torch.utils.dlpack import to_dlpack
import del_fem_dlpack
import numpy as np
import del_msh.TriMesh
import trimesh

def main():
    (tri2vtx, vtx2xyz) = del_msh.TriMesh.sphere()
    print(vtx2xyz[0])
    trimesh.Trimesh(vertices=vtx2xyz, faces=tri2vtx).export("hoge0.obj")
    (vtx2idx, idx2vtx) = del_msh.TriMesh.vtx2vtx(tri2vtx, vtx2xyz.shape[0])
    vtx2idx = torch.from_numpy(vtx2idx.astype(np.uint32)).cuda()
    idx2vtx = torch.from_numpy(idx2vtx.astype(np.uint32)).cuda()
    vtx2xyz_ini = torch.from_numpy(vtx2xyz).cuda()
    vtx2xyz = vtx2xyz_ini.clone()
    vtx2xyz_tmp = vtx2xyz_ini.clone()
    rhs = torch.zeros_like(vtx2xyz)
    # print(vtx2xyz.dtype, vtx2xyz.shape)
    # print(tri2vtx.dtype, vtx2xyz.shape)
    del_fem_dlpack.solve(
       to_dlpack(vtx2idx),
       to_dlpack(idx2vtx),
       to_dlpack(vtx2xyz),
       to_dlpack(rhs),
       1.0,
       to_dlpack(vtx2xyz_tmp))

    print(vtx2xyz_tmp.cpu().numpy()[0])
    print(vtx2xyz.cpu().numpy()[0])
    trimesh.Trimesh(vertices=vtx2xyz.cpu().numpy(), faces=tri2vtx).export("hoge1.obj")



if __name__ == "__main__":
    main()

