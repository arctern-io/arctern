#include "functor/st_point.h"
#include "common/gpu_memory.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
using DataState = GeometryVector::DataState;

struct OutputInfo {
    WkbTag tag;
    int meta_size;
    int value_size;
};

__device__ inline OutputInfo
GetInfoAndDataPerElement(const double* xs,
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
FillInfoKernel(const double* xs, const double* ys, GeoContext results) {
    assert(results.data_state == DataState::FlatOffset_EmptyInfo);
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < results.size) {
        auto out_info = GetInfoAndDataPerElement(xs, ys, index, results, true);
        results.tags[index] = out_info.tag;
        results.meta_offsets[index] = out_info.meta_size;
        results.value_offsets[index] = out_info.value_size;
    }
}


DEVICE_RUNNABLE inline void
AssertInfo(OutputInfo info, const GeoContext& ctx, int index) {
    assert(info.tag.data_ == ctx.get_tag(index).data_);
    assert(info.meta_size == ctx.meta_offsets[index + 1] - ctx.meta_offsets[index]);
    assert(info.value_size == ctx.value_offsets[index + 1] - ctx.value_offsets[index]);
}

static __global__ void
FillDataKernel(const double* xs, const double* ys, GeoContext results) {
    assert(results.data_state == DataState::PrefixSumOffset_EmptyData);
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < results.size) {
        auto out_info = GetInfoAndDataPerElement(xs, ys, index, results);
        AssertInfo(out_info, results, index);
    }
}
}    // namespace

void
ST_Point(const double* cpu_xs, const double* cpu_ys, int size, GeometryVector& results) {
    results.OutputInitialize(size);
    auto xs = GpuMakeUniqueArrayAndCopy(cpu_xs, size);
    auto ys = GpuMakeUniqueArrayAndCopy(cpu_ys, size);
    auto ctx_holder = results.OutputCreateGeoContext();
    {
        auto config = GetKernelExecConfig(size);
        FillInfoKernel<<<config.grid_dim, config.block_dim>>>(
            xs.get(), ys.get(), *ctx_holder);
        ctx_holder->data_state = DataState::FlatOffset_FullInfo;
    }
    results.OutputEvolveWith(*ctx_holder);
    {
        auto config = GetKernelExecConfig(size, 1);
        FillDataKernel<<<config.grid_dim, config.block_dim>>>(
            xs.get(), ys.get(), *ctx_holder);
        ctx_holder->data_state = DataState::PrefixSumOffset_FullData;
    }
    results.OutputFinalizeWith(*ctx_holder);
}

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
