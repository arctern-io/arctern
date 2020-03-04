#include "gis/cuda/common/function_wrapper.h"
#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/conversion/conversions.h"
#include "gis/cuda/conversion/conversions.impl.h"

namespace zilliz {
namespace gis {
namespace cuda {

std::shared_ptr<arrow::Array> ExportWkbFrom(const GeometryVector&);

using internal::WkbArrowContext;

GeometryVector CreateGeometryVectorFromWkb(const std::shared_ptr<arrow::Array>& wkb_) {
  auto wkb = std::static_pointer_cast<arrow::BinaryArray>(wkb_);
  auto size = (int)wkb->length();
  auto binary_bytes = wkb->value_offset(size);
  auto data = GpuMakeUniqueArrayAndCopy((char*)wkb->value_data()->data(), binary_bytes);
  auto offsets = GpuMakeUniqueArrayAndCopy(wkb->raw_value_offsets(), size + 1);
  auto geo_vec = internal::CreateGeometryVectorFromWkbImpl(
      WkbArrowContext{data.get(), offsets.get(), size});
  return geo_vec;
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
