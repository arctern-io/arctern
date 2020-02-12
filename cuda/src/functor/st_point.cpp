#include "functor/st_point.h"

namespace zilliz {
namespace gis {
namespace cpp {

// this is a slow implementation, only to reflex the real proess
enum Action {
    CalculateSpace,
    WriteData
};



//__device__ int size ST_point_kernel(double* meta_output, double* value_output) {
//
//}



__global__ void ST_point_reserve_kernel(const double* xs, const double* ys, int size, GeoContext results) {
    auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < results.size) {

    }
}

void ST_point(const double* xs, const double ys, size_t size, GeometryVector& results) {

}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
