// NOTE:
// this file is template of all functors with a Geometry output
// such as ST_Point, ST_Envelope
#pragma once
#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/functor/geometry_output.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
using DataState = GeometryVector::DataState;

//__device__ inline OutputInfo GetInfoAndDataPerElement(const double* xs, const double*
// ys,
//                                                      int index, GpuContext& results,
//                                                      bool skip_write = false) {
//  if (!skip_write) {
//    auto value = results.get_value_ptr(index);
//    value[0] = xs[index];
//    value[1] = ys[index];
//  }
//  return OutputInfo{WkbTag(WkbCategory::Point, WkbGroup::None), 0, 2};
//}

template <typename Functor>
__global__ void FillInfoKernel(Functor functor, GpuContext results) {
  assert(results.data_state == DataState::FlatOffset_EmptyInfo);
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < results.size) {
    auto out_info = functor(index, results, true);
    results.tags[index] = out_info.tag;
    results.meta_offsets[index] = out_info.meta_size;
    results.value_offsets[index] = out_info.value_size;
  }
}

DEVICE_RUNNABLE inline void AssertInfo(OutputInfo info, const GpuContext& ctx,
                                       int index) {
  assert(info.tag.data == ctx.get_tag(index).data);
  assert(info.meta_size == ctx.meta_offsets[index + 1] - ctx.meta_offsets[index]);
  assert(info.value_size == ctx.value_offsets[index + 1] - ctx.value_offsets[index]);
}

template <typename Functor>
static __global__ void FillDataKernel(Functor functor, GpuContext results) {
  assert(results.data_state == DataState::PrefixSumOffset_EmptyData);
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < results.size) {
    auto out_info = functor(index, results, false);
    AssertInfo(out_info, results, index);
  }
}
}  // namespace

template <typename Functor>
void GeometryOutput(Functor functor, int size, GeometryVector& results) {
  // STEP 1: Initialize vector with size of elements
  results.OutputInitialize(size);
  // STEP 2: Create gpu context according to the vector for cuda
  // where tags and offsets fields are uninitailized
  auto ctx_holder = results.OutputCreateGpuContext();
  {
    // STEP 3: Fill info(tags and offsets) to gpu_ctx using CUDA Kernels
    // where offsets[0, n) is filled with size of each element
    auto config = GetKernelExecConfig(size);
    FillInfoKernel<<<config.grid_dim, config.block_dim>>>(functor, *ctx_holder);
    ctx_holder->data_state = DataState::FlatOffset_FullInfo;
  }
  // STEP 4: Exclusive scan offsets[0, n+1), where offsets[n] = 0
  // then copy info(tags and scanned offsets) back to GeometryVector
  // and alloc cpu & gpu memory for next steps
  results.OutputEvolveWith(*ctx_holder);

  {
    // STEP 5: Fill data(metas and values) to gpu_ctx using CUDA Kernels
    auto config = GetKernelExecConfig(size);
    FillDataKernel<<<config.grid_dim, config.block_dim>>>(functor, *ctx_holder);
    ctx_holder->data_state = DataState::PrefixSumOffset_FullData;
  }
  // STEP 6: Copy data(metas and values) back to GeometryVector
  results.OutputFinalizeWith(*ctx_holder);
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
