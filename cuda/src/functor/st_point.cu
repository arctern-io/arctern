#include "functor/st_point.h"
#include "common/gpu_memory.h"

namespace zilliz {
namespace gis {
namespace cuda {

using DataState = GeometryVector::DataState;
struct OutputInfo {
    WkbTag tag;
    int meta_size;
    int value_size;
};

__device__ inline OutputInfo
ST_point_calc(const double* xs,
              const double* ys,
              int index,
              GeoContext& results,
              bool skip_write = false) {
    if (!skip_write) {
        auto value = results.get_value_ptr(index);
        value[0] = xs[index];
        value[1] = ys[index];
    }
    return OutputInfo{WkbTag(WkbCategory::Point, WkbGroup::None), 0, 2};
}

__global__ void
ST_point_reserve_kernel(const double* xs, const double* ys, GeoContext results) {
    assert(results.data_state == DataState::FlatOffset_EmptyInfo);
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < results.size) {
        auto out_info = ST_point_calc(xs, ys, index, results, true);
        results.tags[index] = out_info.tag;
        results.meta_offsets[index] = out_info.meta_size;
        results.value_offsets[index] = out_info.value_size;
    }
}


DEVICE_RUNNABLE inline void
check_info(OutputInfo info, const GeoContext& ctx, int index) {
    assert(info.tag.data_ == ctx.get_tag(index).data_);
    assert(info.meta_size == ctx.meta_offsets[index + 1] - ctx.meta_offsets[index]);
    assert(info.value_size == ctx.value_offsets[index + 1] - ctx.value_offsets[index]);
}

static __global__ void
ST_point_datafill_kernel(const double* xs, const double* ys, GeoContext results) {
    assert(results.data_state == DataState::PrefixSumOffset_EmptyData);
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < results.size) {
        auto out_info = ST_point_calc(xs, ys, index, results);
        check_info(out_info, results, index);
    }
}


void
ST_point(const double* cpu_xs, const double* cpu_ys, int size, GeometryVector& results) {
    results.OutputInitialize(size);
    auto xs = GpuMakeUniqueArrayAndCopy(cpu_xs, size);
    auto ys = GpuMakeUniqueArrayAndCopy(cpu_ys, size);
    auto ctx = results.OutputCreateGeoContext();
    {
        auto config = GetKernelExecConfig(size);
        ST_point_reserve_kernel<<<config.grid_dim, config.block_dim>>>(
            xs.get(), ys.get(), ctx.get());
        ctx->data_state = DataState::FlatOffset_FullInfo;
    }
    results.OutputEvolveWith(ctx.get());
    {
        auto config = GetKernelExecConfig(size, 1);
        ST_point_datafill_kernel<<<config.grid_dim, config.block_dim>>>(
            xs.get(), ys.get(), ctx.get());
        ctx->data_state = DataState::PrefixSumOffset_FullData;
    }
    results.OutputFinalizeWith(ctx.get());
}

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
