#include <exception>

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/conversion/conversions.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace internal {
__global__ static void CalcOffsets(GpuContext input, WkbArrowContext output, int size) {
  auto index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < size) {
    auto common_offset = (int)((sizeof(WkbByteOrder) + sizeof(WkbTag)) * index);
    auto value_offset = input.value_offsets[index] * sizeof(double);
    auto meta_offset = input.meta_offsets[index] * sizeof(int);
    auto wkb_length = common_offset + value_offset + meta_offset;
    output.offsets[index] = wkb_length;
  }
}

// return: size of total data length in bytes
void ToArrowWkbFillOffsets(const GpuContext& input, WkbArrowContext& output,
                          int* value_length_ptr) {
  assert(input.size == output.size);
  assert(output.offsets);
  assert(!output.values);
  {
    auto offset_size = input.size + 1;
    auto config = GetKernelExecConfig(offset_size);
    CalcOffsets<<<config.grid_dim, config.block_dim>>>(input, output, offset_size);
  }
  if (value_length_ptr) {
    auto src = output.offsets + input.size;
    GpuMemcpy(value_length_ptr, src, 1);
  }
}

struct WkbEncoder {
  const uint32_t* metas;
  const double* values;
  char* wkb_iter;
  static constexpr bool skip_write = false;

 private:
  __device__ void ValuesToWkb(int demensions, int points) {
    auto count = demensions * points;
    auto bytes = count * sizeof(double);
    if (!skip_write) {
      memcpy(wkb_iter, values, bytes);
    }
    wkb_iter += bytes;
    values += count;
  }

  template <typename T>
  __device__ T MetaToWkb() {
    static_assert(sizeof(T) == sizeof(*metas), "size of T must match meta");
    auto m = static_cast<T>(*metas);
    if (!skip_write) {
      InsertIntoWkb(m);
    }
    metas += 1;
    return m;
  }

 public:
  template <typename T>
  __device__ void InsertIntoWkb(T data) {
    int len = sizeof(T);
    memcpy(wkb_iter, &data, len);
    wkb_iter += len;
  }

  __device__ void EncodePoint(int demensions) {
    // wtf
    ValuesToWkb(demensions, 1);
  }

  __device__ void EncodeLineString(int demensions) {
    auto size = MetaToWkb<int>();
    ValuesToWkb(demensions, size);
  }

  __device__ void EncodePolygon(int demensions) {
    auto polys = MetaToWkb<int>();
    for (int i = 0; i < polys; ++i) {
      EncodeLineString(demensions);
    }
  }
};

__global__ static void CalcValues(const GpuContext input, WkbArrowContext output) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < input.size) {
    auto tag = input.get_tag(index);
    assert(tag.get_group() == WkbGroup::None);
    int dimensions = 2;
    auto metas = input.get_meta_ptr(index);
    auto values = input.get_value_ptr(index);
    auto wkb_iter = output.get_wkb_ptr(index);

    WkbEncoder encoder{metas, values, wkb_iter};
    encoder.InsertIntoWkb(WkbByteOrder::LittleEndian);
    encoder.InsertIntoWkb(tag);

    switch (tag.get_category()) {
      case WkbCategory::Point: {
        encoder.EncodePoint(dimensions);
        break;
      }
      case WkbCategory::LineString: {
        encoder.EncodeLineString(dimensions);
        break;
      }
      case WkbCategory::Polygon: {
        encoder.EncodePolygon(dimensions);
        break;
      }
      default: {
        assert(false);
        break;
      }
    }
    auto wkb_length = encoder.wkb_iter - wkb_iter;
    auto std_wkb_length =  output.get_wkb_ptr(index + 1) - output.get_wkb_ptr(index);
    assert(std_wkb_length == wkb_length);
  }
}

void ToArrowWkbFillValues(const GpuContext& input, WkbArrowContext& output) {
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
