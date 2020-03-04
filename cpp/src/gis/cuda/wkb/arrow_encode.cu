#include <exception>

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/wkb/conversions.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace internal {
__global__ static void CalcOffsets(GpuContext input, WkbArrowContext output, int size) {
  auto index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < size) {
    auto value_offset = input.value_offsets[index];
    auto meta_offset = input.meta_offsets[index];
    auto wkb_length = sizeof(WkbByteOrder) + value_offset + meta_offset;
    output.offsets[index] = wkb_length;
  }
}

// return: size of total data length in bytes
void ExportWkbFillOffsets(const GpuContext& input, WkbArrowContext& output,
                          int* value_length) {
  assert(input.size == output.size);
  assert(output.offsets);
  assert(!output.values);
  {
    auto offset_size = input.size + 1;
    auto config = GetKernelExecConfig(offset_size);
    CalcOffsets<<<config.grid_dim, config.block_dim>>>(input, output, offset_size);
  }
  if (value_length) {
    auto src = input.meta_offsets + input.size;
    GpuMemcpy(value_length, src, 1);
  }
}

__global__ static void CalcValues(GpuContext input, WkbArrowContext output) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < input.size) {
    auto tag = input.get_tag(index);
    assert(tag.get_group() == WkbGroup::None);

    switch (tag.get_category()) {
      default: {
        break;
      }
    }
  }
}

void ExportWkbFillValues(const GpuContext& input, WkbArrowContext& output) {
  assert(input.size == output.size);
  assert(output.offsets);
  assert(output.values);
  auto config = GetKernelExecConfig(input.size);
  CalcValues<<<config.grid_dim, config.block_dim>>>(input, output);
}

}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
