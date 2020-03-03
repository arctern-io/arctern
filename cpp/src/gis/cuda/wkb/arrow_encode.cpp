#include <exception>

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/wkb/conversions.h"

namespace zilliz {
namespace gis {
namespace cuda {
using internal::WkbArrowContext;

namespace {
void check(arrow::Status status) {
  if (!status.ok()) {
    throw std::runtime_error("action failed for " + status.message());
  }
}

template <typename T>
std::shared_ptr<arrow::Buffer> AllocArrowBufferAndCopy(int size, const T* dev_ptr) {
  std::shared_ptr<arrow::Buffer> buffer;
  auto len = sizeof(T) * size;
  auto status = arrow::AllocateBuffer(len, &buffer);
  check(status);
  GpuMemcpy((T*)buffer->mutable_data(), dev_ptr, size);
  return buffer;
}
}  // namespace

std::shared_ptr<arrow::Array> ExportWkbFrom(const GeometryVector& geo_vec) {
  auto size = geo_vec.size();
  auto offsets = GpuMakeUniqueArray<int>(size);

  WkbArrowContext arrow_ctx{nullptr, offsets.get(), size};
  auto values_length = internal::ExportWkbFillOffsets(geo_vec, arrow_ctx);
  auto values = GpuMakeUniqueArray<char>(values_length);
  internal::ExportWkbFillValues(geo_vec, arrow_ctx);

  auto offsets_buffer = AllocArrowBufferAndCopy(size + 1, offsets.get());
  auto values_buffer = AllocArrowBufferAndCopy(values_length, offsets.get());

  auto results = std::make_shared<arrow::BinaryArray>(size, offsets_buffer, values_buffer,
                                                      nullptr, 0);
  return results;
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
