#include "functor/st_point.h"

namespace zilliz {
namespace gis {
namespace cuda {

constexpr int max_buffer_per_meta = 50;
constexpr int max_buffer_per_value = 250;
using DataState = GeometryVector::DataState;


struct OutputInfo {
    WkbTag tag;
    int meta_size;
    int value_size;
};

__device__ inline OutputInfo
ST_point_compute_kernel(const double* xs,
                        const double* ys,
                        int index,
                        uint32_t* meta_output,
                        double* value_output,
                        bool skip_write = false) {
    (void)meta_output;
    if (!skip_write) {
        assert(value_output != nullptr);
        value_output[0] = xs[index];
        value_output[1] = ys[index];
    }
    return OutputInfo{WkbTag(WkbCategory::Point, WkbGroup::None), 0, 2};
}

__global__ void
ST_point_reserve_kernel(const double* xs, const double* ys, GeoContext results) {
    assert(results.data_state == DataState::FlatOffset_EmptyInfo);
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < results.size) {
        auto out_info = ST_point_compute_kernel(xs, ys, index, nullptr, nullptr, true);
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

static
__global__ void
ST_point_datafill_kernel(const double* xs,
                         const double* ys,
                         int size,
                         GeoContext results,
                         GeoWorkspace) {
    assert(results.data_state == DataState::PrefixSumOffset_EmptyData);
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < results.size) {
        auto meta_output = results.get_meta_ptr(index);
        auto value_output = results.get_value_ptr(index);
        auto out_info = ST_point_compute_kernel(xs, ys, index, meta_output, value_output);
        check_info(out_info, results, index);
    }
}


// GeometryVector
// ST_point(const double* xs, const double ys, size_t size) {
//
//}

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
