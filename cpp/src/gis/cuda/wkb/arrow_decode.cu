#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/functor/geometry_output.h"
#include "gis/cuda/wkb/conversions.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
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

__device__ inline OutputInfo GetInfoAndDataPerElement(const WkbArrowContext& input,
                                                      int index, GpuContext& results,
                                                      bool skip_write) {
  auto wkb_iter = input.get_wkb_ptr(index);
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
}  // namespace

namespace internal {
GeometryVector CreateGeometryVectorFromWkbImpl(WkbArrowContext input) {
  GeometryVector results;
  int size = input.size;
  // TODO(dog): add hanlder for nulls
  assert(input.null_counts() == 0);
  auto functor = [input](int index, GpuContext& results, bool skip_write) -> OutputInfo {
    return GetInfoAndDataPerElement(input, index, results, skip_write);
  };
  GeometryOutput(functor, size, results);
  return results;
}

}  // namespace internal
}  // namespace cuda
}  // namespace gis
}  // namespace zilliz