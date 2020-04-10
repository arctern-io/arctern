#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_relate.h"
#include "gis/cuda/tools/relations.h"

namespace arctern {
namespace gis {
namespace cuda {

using ConstIter = ConstGpuContext::ConstIter;
using de9im::Matrix;
__device__ Matrix PointRelateTo(ConstIter& left_iter, const ConstGpuContext& right,
                                Matrix matrix, int index) {
  (void)matrix;  // ignore
  auto right_tag = right.get_tag(index);
  assert(right_tag.get_space_type() == WkbSpaceType::XY);
  auto left_point = *(const double2*)left_iter.values;
  left_iter.values += 2;

  auto right_iter = right.get_iter(index);
  Matrix result;
  switch (right_tag.get_category()) {
    case WkbCategory::kPoint: {
      auto right_point = *(const double2*)right_iter.values;
      right_iter.values += 2;
      auto is_eq = IsEqual(left_point, right_point);
      result = is_eq ? Matrix("0FFFFFFF*") : Matrix("FF0FFF0F*");
      break;
    }
    case WkbCategory::kLineString: {
      auto size = (int)*right_iter.metas++;
      auto points = (const double2*)right_iter.values;

      if (size == 0) {
        result = Matrix("FFFFFFFF*");
        break;
      }

      if(size == 1) {
        auto right_point = points[0];
        right_iter.values += 2;
        auto is_eq = IsEqual(left_point, right_point);
        result = is_eq ? Matrix("F0FFFFF0*") : Matrix("FF0FFFF0*");
        break;
      }


      assert(size >= 2);
      auto endpoint0 = points[0];
      auto endpoint1 = points[size - 1];
      Matrix mat;

      using Position = Matrix::Position;
      using State = Matrix::State;

      if(IsEqual(endpoint0, left_point) || IsEqual(endpoint1, left_point)) {
        mat.set_col<Position::kBorderline>("0FF");
      } else {
        mat.set_col<Position::kBorderline>("FF0");
      }
      bool point_in_line = false;
      for (int i = 0; i < size - 1; ++i) {
        auto u = points[i];
        auto v = points[i + 1];
        if(i != 0 && IsEqual(u, left_point)) {
          point_in_line = true;
          break;
        }
        is_in_line(u, )
      }

    }
  }
}

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
