//
// Created by mike on 2/10/20.
//
#include "wkb/gis_definitions.h"
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

namespace zilliz {
namespace gis {
namespace cpp {

using GeoContext = GeometryVector::GPUContext;

__global__
void ST_distance_kernel(GeoContext left, GeoContext right, double* result) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < left.size) {
        
    }
}

void
ST_distance(const GeometryVector& left, const GeometryVector& right, double* result) {

}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
