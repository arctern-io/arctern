#include "gis/cuda/common/gis_definitions.h"
namespace zilliz {
namespace gis {
namespace cuda {



std::shared_ptr<arrow::Array> GeometryVector::ExportToArrowWkb() {
  // TODO(dog)
  return nullptr;
}

namespace {
using DataState = GeometryVector::DataState;

struct OutputInfo {
  WkbTag tag;
  int meta_size;
  int value_size;
};

__device__ inline OutputInfo GetInfoAndDataPerElement(const GpuContext& input, int index,
                                                      GpuContext& results,
                                                      bool skip_write = false) {
  if (!skip_write) {

  }
  auto result_tag = WkbTag(WkbCategory::Polygon, WkbGroup::None);
  return OutputInfo{result_tag, 1 + 1, 2 * 5};
}

__global__ void FillInfoKernel(const GpuContext input, GpuContext results) {
  assert(results.data_state == DataState::FlatOffset_EmptyInfo);
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  assert(input.size == results.size);
  if (index < input.size) {
    auto out_info = GetInfoAndDataPerElement(input, index, results, true);
    printf("%d", index);
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

static __global__ void FillDataKernel(const GpuContext input, GpuContext results) {
  assert(results.data_state == DataState::PrefixSumOffset_EmptyData);
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < results.size) {
    auto out_info = GetInfoAndDataPerElement(input, index, results);
    AssertInfo(out_info, results, index);
  }
}
}


namespace GeometryVectorFactory {
GeometryVector CreateFromArrowWkb(std::shared_ptr<arrow::Array> wkb) {
  GeometryVector results;
  int size = wkb->length();
  // TODO(dog): add hanlder for nulls
  assert(wkb->null_count() == 0);

  // STEP 1: Initialize vector with size of elements
  results.OutputInitialize(size);

  // STEP 2: Create gpu context according to the vector for cuda
  // where tags and offsets fields are uninitailized
  auto ctx_holder = results.OutputCreateGpuContext();
  {
    // STEP 3: Fill info(tags and offsets) to gpu_ctx using CUDA Kernels
    // where offsets[0, n) is filled with size of each element
    auto config = GetKernelExecConfig(size);
    FillInfoKernel<<<config.grid_dim, config.block_dim>>>(xs.get(), ys.get(),
        *ctx_holder);
    ctx_holder->data_state = DataState::FlatOffset_FullInfo;
  }

  // STEP 4: Exclusive scan offsets[0, n+1), where offsets[n] = 0
  // then copy info(tags and scanned offsets) back to GeometryVector
  // and alloc cpu & gpu memory for next steps
  results.OutputEvolveWith(*ctx_holder);

  {
    // STEP 5: Fill data(metas and values) to gpu_ctx using CUDA Kernels
    auto config = GetKernelExecConfig(size);
    FillDataKernel<<<config.grid_dim, config.block_dim>>>(xs.get(), ys.get(),
        *ctx_holder);
    ctx_holder->data_state = DataState::PrefixSumOffset_FullData;
  }
  // STEP 6: Copy data(metas and values) back to GeometryVector
  results.OutputFinalizeWith(*ctx_holder);

  return results;
}

}  // namespace GeometryVectorFactory
}  // namespace cuda
}  // namespace gis
}  // namespace zilliz