#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_relate.h"
#include "gis/cuda/tools/relations.h"

namespace arctern {
namespace gis {
namespace cuda {

using ConstIter = ConstGpuContext::ConstIter;
using de9im::Matrix;
namespace {
__device__ Matrix PointRelateTo(ConstIter& left_iter, const ConstGpuContext& right,
                                Matrix matrix, int index) {
  (void)matrix;  // ignore
  auto right_tag = right.get_tag(index);
  assert(right_tag.get_space_type() == WkbSpaceType::XY);
  auto left_point = *(const double2*)left_iter.values;
  left_iter.values += 2;

  auto right_iter = right.get_iter(index);
  switch (right_tag.get_category()) {
    case WkbCategory::kPoint: {
      auto right_point = *(const double2*)right_iter.values;
      right_iter.values += 2;
      auto is_eq = is_equal(left_point, right_point);
      auto result = is_eq ? Matrix("0FFFFFFF*") : Matrix("FF0FFF0F");
      break;
    }
    case WkbCategory::kLineString: {
      auto size = (int)*right_iter.metas++;
      auto values2 = (const double2*)right_iter.values;

      if(size == 0) {
        return
      }
      for(int i = 0; i < size - 1; ++i) {
        auto point = values2[i];

      }
    }
  }
}
}  // namespace

__device__ Matrix RelateOp(ConstGpuContext& left, ConstGpuContext& right,
                           de9im::Matrix input_matrix, int index) {
  auto left_tag = left.get_tag(index);
  assert(left_tag.get_space_type() == WkbSpaceType::XY);
  de9im::Matrix result;
  auto left_iter = left.get_iter(index);
  switch (left_tag.get_category()) {
    case WkbCategory::kPoint: {
      result = PointRelateTo(left_iter, right, input_matrix, index);
      break;
    }
    case WkbCategory::kMultiLineString: {
      // TODO
      result = de9im::INVALID_MATRIX;
      break;
    }
    default: {
      result = de9im::INVALID_MATRIX;
      left_iter = left.get_iter(index + 1);
      break;
    }
  }
  assert(left_iter.values == right.get_value_ptr(index));
  assert(left_iter.metas == right.get_meta_ptr(index));
  return result;
}

static __global__ void ST_RelateImpl(ConstGpuContext left, ConstGpuContext right,
                                     de9im::Matrix input_matrix,
                                     de9im::Matrix* output_matrixes) {
  //
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < left.size) {
    output_matrixes[index] = RelateOp(left, right, input_matrix, index);
    // TODO: check hint, use bool api
  }
}

void ST_Relate(const GeometryVector& left_vec, const GeometryVector& right_vec,
               de9im::Matrix input_matrix, de9im::Matrix* host_output_matrixes) {
  assert(left_vec.size() == right_vec.size());
  auto size = left_vec.size();
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
