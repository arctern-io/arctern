//
// Created by mike on 2/10/20.
//

#include "functor/st_area.h"
#include "common/gpu_memory.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
inline DEVICE_RUNNABLE double
PolygonArea(const GpuContext& ctx, int index) {
    auto meta = ctx.get_meta_ptr(index);
    auto value = ctx.get_value_ptr(index);
    assert(meta[0] == 1);
    auto count = (int)meta[1];
    double sum_area = 0;
    for (int point_index = 0; point_index < count; ++point_index) {
        auto lv = value + 2 * point_index;
        auto rv = (point_index + 1 == count) ? value : lv + 2;
        auto area = lv[0] * rv[1] - lv[1] * rv[0];
        sum_area += area;
    }
    return fabs(sum_area / 2);
}


__global__ void
ST_AreaKernel(GpuContext ctx, double* result) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < ctx.size) {
        auto tag = ctx.get_tag(tid);
        // handle 2d case only for now
        assert(tag.get_group() == WkbGroup::None);
        switch (tag.get_category()) {
            case WkbCategory::Polygon: {
                result[tid] = PolygonArea(ctx, tid);
                break;
            }
            default: {
                assert(false);
            }
        }
    }
}
}    // namespace

void
ST_Area(const GeometryVector& vec, double* host_results) {
    auto ctx_holder = vec.CreateReadGpuContext();
    auto config = GetKernelExecConfig(vec.size());
    auto dev_result = GpuMakeUniqueArray<double>(vec.size());
    ST_AreaKernel<<<config.grid_dim, config.block_dim>>>(*ctx_holder, dev_result.get());
    GpuMemcpy(host_results, dev_result.get(), vec.size());
}

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
