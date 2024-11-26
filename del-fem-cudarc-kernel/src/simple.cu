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

__global__
void solve_diffuse3_jacobi(
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
    float rhs[3] = {
        vtx2rhs[i_vtx*3+0],
        vtx2rhs[i_vtx*3+1],
        vtx2rhs[i_vtx*3+2] };
    for(uint32_t idx = vtx2idx[i_vtx]; idx < vtx2idx[i_vtx+1]; ++idx ) {
        uint32_t j_vtx = idx2vtx[idx];
        rhs[0] += vtx2lhs_ini[j_vtx*3+0];
        rhs[1] += vtx2lhs_ini[j_vtx*3+1];
        rhs[2] += vtx2lhs_ini[j_vtx*3+2];
    }
    const float dtmp = float(vtx2idx[i_vtx+1] - vtx2idx[i_vtx]) + lambda;
    vtx2lhs_upd[i_vtx*3+0] = rhs[0] / dtmp;
    vtx2lhs_upd[i_vtx*3+1] = rhs[1] / dtmp;
    vtx2lhs_upd[i_vtx*3+2] = rhs[2] / dtmp;
    float r0 = rhs[0] - vtx2lhs_ini[i_vtx*3+0] * dtmp;
    float r1 = rhs[1] - vtx2lhs_ini[i_vtx*3+1] * dtmp;
    float r2 = rhs[2] - vtx2lhs_ini[i_vtx*3+2] * dtmp;
    vtx2res[i_vtx] = r0*r0 + r1*r1 + r2*r2;
}

}

