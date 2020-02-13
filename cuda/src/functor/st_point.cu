#include "functor/st_point.h"

namespace zilliz {
namespace gis {
namespace cuda {

constexpr int max_buffer_per_meta = 50;
constexpr int max_buffer_per_value = 250;
using DataState = GeometryVector::DataState;


struct OutputMessage {
    WkbTag tag;
    int meta_size;
    int value_size;
};

__device__ inline OutputMessage
ST_point_compute_kernel(const double* xs,
                        const double* ys,
                        int index,
                        uint32_t* meta_output,
                        double* value_output) {
    (void)meta_output;
    value_output[0] = xs[index];
    value_output[1] = ys[index];
    return OutputMessage{WkbTag(WkbCategory::Point, WkbGroup::None), 0, 2};
}

__global__ void
ST_point_reserve_kernel(const double* xs,
                        const double* ys,
                        GeoContext results) {
    assert(results.data_state == DataState::FlatOffset_EmptyData);
    auto init_index = threadIdx.x + blockIdx.x * blockDim.x;
    auto meta_buffer = results.metas + max_buffer_per_meta * init_index;
    auto value_buffer = results.values + max_buffer_per_value * init_index;
    for (auto index = init_index; index < results.size; index += blockDim.x * gridDim.x) {
        auto out_message = ST_point_compute_kernel(xs, ys, index, meta_buffer, value_buffer);
        results.tags[index] = out_message.tag;
        results.meta_offsets[index] = out_message.meta_size;
        results.value_offsets[index] = out_message.value_size;
    }
}

__global__ void
ST_point_datafill_kernel(const double* xs,
                        const double* ys,
                        int size,
                        GeoContext results) {
    assert(results.data_state == DataState::PrefixSumOffset_EmptyData);
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < results.size) {
        auto meta_output = results.get_meta_ptr(index);
        auto value_output = results.get_value_ptr(index);
    }
}



void
ST_point(const double* xs, const double ys, size_t size, GeometryVector& results) {}

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
