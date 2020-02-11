//
// Created by mike on 2/10/20.
//
#include "wkb/gis_definitions.h"
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cmath>

namespace zilliz {
namespace gis {
namespace cpp {

using GeoContext = GeometryVector::GPUContext;

inline
DEVICE_RUNNABLE double ST_distance_point_point(const GeoContext& left, const GeoContext& right, int index) {
    auto lv = left.get_value_ptr(index);
    auto rv = right.get_value_ptr(index);
    auto dx = (lv[0] - rv[0]);
    auto dy = (lv[1] - rv[1]);
    return sqrt(dx * dx + dy * dy);
}

__global__
void ST_distance_kernel(GeoContext left, GeoContext right, double* result) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < left.size) {
        auto left_tag = left.get_tag(tid);
        auto right_tag = right.get_tag(tid);
        assert(left_tag.get_group() == WKB_Group::None);
        assert(right_tag.get_group() == WKB_Group::None);
        if(left_tag.get_category() == WKB_Category::Point && right_tag.get_category() == WKB_Category::Point) {
            result[tid] = ST_distance_point_point(left, right, tid);
        } else {
            result[tid] = NAN;
        }
    }
}

void
ST_distance(const GeometryVector& left, const GeometryVector& right, double* result) {

}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
