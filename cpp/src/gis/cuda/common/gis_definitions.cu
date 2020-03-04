#include <thrust/scan.h>
namespace zilliz {
namespace gis {
namespace cuda {
namespace internal {

void ExclusiveScan(int* offsets, int size) {
  // thrust library must be used in *.cu files
  thrust::exclusive_scan(thrust::cuda::par, offsets, offsets + size, offsets, 0);
}

}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
