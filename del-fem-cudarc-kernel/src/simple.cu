#include <stdio.h>
#include <cuda_runtime.h>

extern "C" {

__global__
void solve_diffuse_jacobi(
    const uint32_t num_vtx,
    const uint32_t *vtx2idx,
    const uint32_t *idx2vtx,
    float lambda,
    const float *vtx2rhs,
    float *vtx2lhs_ini,
    float *vtx2lhs_upd,
    float *vtx2res)
{
    int i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_vtx >= num_vtx) { return; }
    //
    float rhs = vtx2rhs[i_vtx];
    for(uint32_t idx = vtx2idx[i_vtx]; idx < vtx2idx[i_vtx+1]; ++idx ) {
        uint32_t j_vtx = idx2vtx[idx];
        rhs += vtx2lhs_ini[j_vtx];
    }
    const float dtmp = float(vtx2idx[i_vtx+1] - vtx2idx[i_vtx]) + lambda;
    vtx2lhs_upd[i_vtx] = rhs / dtmp;
    vtx2res[i_vtx] = rhs - vtx2lhs_ini[i_vtx] * dtmp;
}

}

