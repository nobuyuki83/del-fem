
#include "edge2.h"

namespace spring2 {

struct ReturnPbd {
  cuda::std::array<float,2> dp0;
  cuda::std::array<float,2> dp1;
};

__device__
auto pbd(
    const float* p0_def,
    const float* p1_def,
    float len_ini,
    float w0,
    float w1) -> ReturnPbd
{
    const float len_def = edge2::length(p0_def, p1_def);
    const auto e01 = edge2::unit_edge_vector(p0_def, p1_def);
    return ReturnPbd {
        {
            w0 / (w0 + w1) * (len_def - len_ini) * e01[0],
            w0 / (w0 + w1) * (len_def - len_ini) * e01[1],
        },
        {
            -w1 / (w0 + w1) * (len_def - len_ini) * e01[0],
            -w1 / (w0 + w1) * (len_def - len_ini) * e01[1],
        }
    };
}


}