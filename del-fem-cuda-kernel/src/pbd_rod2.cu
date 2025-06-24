#include <cstdint> // for uint32_t
#include <stdio.h>
#include <cuda_runtime.h>

#include "edge2.h"
#include "spring2.h"

extern "C" {

__global__
void solve(
    const uint32_t num_example,
    const uint32_t num_point,
    const float *pnt2xyini,
    const float *pnt2massinv,
    float dt,
    float *gravity,
    float *example2pnt2xydef,
    float *example2pnt2xynew,
    float *example2pnt2velo)
{
    int i_example = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_example >= num_example) { return; }
    // ---------------
    float* pnt2xydef = example2pnt2xydef + i_example * num_point * 2;
    float* pnt2xynew = example2pnt2xynew + i_example * num_point * 2;
    float* pnt2velo = example2pnt2velo + i_example * num_point * 2;
    for(int i_point=0;i_point<num_point;++i_point) {
      pnt2xydef[i_point*2+0] = pnt2xyini[i_point*2+0];
      pnt2xydef[i_point*2+1] = pnt2xyini[i_point*2+1];
      pnt2velo[i_point*2+0] = 0.f;
      pnt2velo[i_point*2+1] = 0.f;
    }
    for(int i_step = 0; i_step<1000; ++i_step) {
        for(int i_point = 0; i_point < num_point; ++i_point) {
            if( pnt2massinv[i_point] == 0.f ){
                continue;
            }
            pnt2xynew[i_point * 2 + 0] =
                pnt2xydef[i_point * 2 + 0] + dt * dt * gravity[0] + dt * pnt2velo[i_point * 2 + 0];
            pnt2xynew[i_point * 2 + 1] =
                pnt2xydef[i_point * 2 + 1] + dt * dt * gravity[1] + dt * pnt2velo[i_point * 2 + 1];
        }
        for(int i_seg = 0; i_seg < num_point - 1; ++i_seg) {
            int ip0 = i_seg;
            int ip1 = i_seg + 1;
            const float* p0_def = pnt2xynew + ip0 * 2;
            const float* p1_def = pnt2xynew + ip1 * 2;
            const float* p0_ini = pnt2xyini + ip0 * 2;
            const float* p1_ini = pnt2xyini + ip1 * 2;
            const float w0 = pnt2massinv[ip0];
            const float w1 = pnt2massinv[ip1];
            const float len_ini = edge2::length(p0_ini, p1_ini);
            const spring2::ReturnPbd r = spring2::pbd(p0_def, p1_def, len_ini, w0, w1);
            pnt2xynew[ip0 * 2 + 0] += r.dp0[0];
            pnt2xynew[ip0 * 2 + 1] += r.dp0[1];
            pnt2xynew[ip1 * 2 + 0] += r.dp1[0];
            pnt2xynew[ip1 * 2 + 1] += r.dp1[1];
        }
        for(int i_point = 0; i_point < num_point; ++i_point) {
            pnt2velo[i_point] = (pnt2xynew[i_point] - pnt2xydef[i_point]) / dt;
            pnt2xydef[i_point] = pnt2xynew[i_point];
        }
    }
}

}