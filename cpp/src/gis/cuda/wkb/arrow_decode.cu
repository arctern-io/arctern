#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/wkb/conversions.h"
#include "gis/cuda/functor/geometry_output.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
using DataState = GeometryVector::DataState;

struct Iter {
  uint32_t* metas;
  double* values;
};

struct WkbDecoder {
  const char* wkb_iter;
  uint32_t* metas;
  double* values;
  bool skip_write;

 public:
  template <typename T>
  __device__ T fetch() {
    T tmp;
    int len = sizeof(T);
    memcpy(&tmp, wkb_iter, len);
    wkb_iter += len;
    return tmp;
  }

  __device__ void extend_values(int demensions, int points) {
    auto count = demensions * points;
    auto bytes = count * sizeof(double);
    if (!skip_write) {
      memcpy(&values, wkb_iter, bytes);
    }
    wkb_iter += bytes;
    values += count;
  }

  __device__ int extend_size_meta() {
    auto size = fetch<int>();
    if (!skip_write) {
      *metas = size;
    }
    metas += 1;
    return size;
  }

 public:
  __device__ void DecodePoint(int demensions) { extend_values(demensions, 1); }

  __device__ void DecodeLineString(int demensions) {
    auto size = extend_size_meta();
    extend_values(demensions, size);
  }

  __device__ void DecodePolygon(int demensions) {
    auto polys = extend_size_meta();
    for (int i = 0; i < polys; ++i) {
      auto size = extend_size_meta();
      extend_values(demensions, size);
    }
  }
};

__device__ inline OutputInfo GetInfoAndDataPerElement(const WkbArrowContext& ctx, int index,
                                                      GpuContext& results,
                                                      bool skip_write = false) {
  auto wkb_iter = ctx.get_wkb_ptr(index);
  auto metas = results.get_meta_ptr(index);
  auto values = results.get_value_ptr(index);

  WkbDecoder decoder{wkb_iter, metas, values, skip_write};

  auto byte_order = decoder.fetch<WkbByteOrder>();
  assert(byte_order == WkbByteOrder::LittleEndian);
  auto tag = decoder.fetch<WkbTag>();
  assert(tag.get_group() == WkbGroup::None);
  constexpr auto demensions = 2;
  switch (tag.get_category()) {
    case WkbCategory::Point: {
      decoder.DecodePoint(demensions);
      break;
    }
    case WkbCategory::LineString: {
      decoder.DecodeLineString(demensions);
      break;
    }
    case WkbCategory::Polygon: {
      decoder.DecodePolygon(demensions);
      break;
    }
    default: {
      break;
    }
  }

  auto result_tag = WkbTag(WkbCategory::Polygon, WkbGroup::None);
  return OutputInfo{result_tag, 1 + 1, 2 * 5};
}

__global__ void FillInfoKernel(const WkbArrowContext input, GpuContext results) {
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

__global__ void FillDataKernel(const WkbArrowContext input, GpuContext results) {
  assert(results.data_state == DataState::PrefixSumOffset_EmptyData);
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < results.size) {
    auto out_info = GetInfoAndDataPerElement(input, index, results);
    AssertInfo(out_info, results, index);
  }
}
}  // namespace

namespace GeometryVectorFactoryImpl {
GeometryVector CreateFromArrowWkbContextImpl(WkbArrowContext input) {
  GeometryVector results;
  int size = input.size;
  // TODO(dog): add hanlder for nulls
  assert(input.null_counts() == 0);

  // STEP 1: Initialize vector with size of elements
  results.OutputInitialize(size);

  // STEP 2: Create gpu context according to the vector for cuda
  // where tags and offsets fields are uninitailized
  auto ctx_holder = results.OutputCreateGpuContext();
  {
    // STEP 3: Fill info(tags and offsets) to gpu_ctx using CUDA Kernels
    // where offsets[0, n) is filled with size of each element
    auto config = GetKernelExecConfig(size);
    FillInfoKernel<<<config.grid_dim, config.block_dim>>>(input, *ctx_holder);
    ctx_holder->data_state = DataState::FlatOffset_FullInfo;
  }

  // STEP 4: Exclusive scan offsets[0, n+1), where offsets[n] = 0
  // then copy info(tags and scanned offsets) back to GeometryVector
  // and alloc cpu & gpu memory for next steps
  results.OutputEvolveWith(*ctx_holder);

  {
    // STEP 5: Fill data(metas and values) to gpu_ctx using CUDA Kernels
    auto config = GetKernelExecConfig(size);
    FillDataKernel<<<config.grid_dim, config.block_dim>>>(input, *ctx_holder);
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