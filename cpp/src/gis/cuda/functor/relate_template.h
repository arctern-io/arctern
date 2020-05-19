#pragma once
#include "gis/cuda/functor/st_relate.h"

namespace arctern {
namespace gis {
namespace cuda {

inline auto GenRelateMatrix(const ConstGpuContext& left_ctx,
                            const ConstGpuContext& right_ctx) {
  auto len = left_ctx.size;
  auto result = GpuMakeUniqueArray<de9im::Matrix>(len);
  ST_Relate(left_ctx, right_ctx, result.get());
  return result;
}

namespace internal {
template <typename Func>
__global__ void RelationFinalizeImpl(Func func, const de9im::Matrix* dev_matrices,
                                     int64_t size, bool* dev_results) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    dev_results[index] = func(dev_matrices[index]);
  }
}
}  // namespace internal

template <typename Func>
void RelationFinalize(Func func, const de9im::Matrix* dev_matrices, int64_t size,
                      bool* dev_results) {
  auto config = GetKernelExecConfig(size);
  internal::RelationFinalizeImpl<<<config.grid_dim, config.block_dim>>>(
      func, dev_matrices, size, dev_results);
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
