#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_relate.h"
#include "gis/cuda/tools/relation.h"

namespace arctern {
namespace gis {
namespace cuda {

using ConstIter = ConstGpuContext::ConstIter;
using de9im::Matrix;

__device__ Matrix PointRelateToLineString(double2 left_point, int right_size,
                                          const double2* right_points) {
  if (right_size == 0) {
    return Matrix("FFFFFFFF*");
  }

  //  if (right_size == 1) {
  //    auto right_point = right_points[0];
  //    auto is_eq = IsEqual(left_point, right_point);
  //    return is_eq ? Matrix("F0FFFFF0*") : Matrix("FF0FFFF0*");
  //  }

  assert(right_size >= 2);
  Matrix mat;

  using Position = Matrix::Position;
  using State = Matrix::State;

  auto point_on_line = PointOnInnerLineString(left_point, right_size, right_points);
  if (point_on_line) {
    mat.set_col<Position::kInterier>("0F0");
    mat.set_col<Position::kExterier>("FF*");
  } else {
    mat.set_col<Position::kInterier>("FF0");
    // false positive: boundary is not included in exterier
    mat.set_col<Position::kExterier>("0F*");
  }
  // endpoints
  auto ep0 = right_points[0];
  auto ep1 = right_points[right_size - 1];
  if (IsEqual(ep0, left_point) || IsEqual(ep1, left_point)) {
    mat.set_col<Position::kBoundry>("0FF");
    // fix possible false positive
    mat.set<Position::kInterier, Position::kExterier>(State::kFalse);
  } else {
    mat.set_col<Position::kBoundry>("FF0");
  }
  return mat;
}

__device__ Matrix LineStringRelateToLineString(int size, ConstIter& left_iter,
                                               ConstIter& right_iter) {
  //

  auto left_size = left_iter.read_meta<int>();
  auto left_points = left_iter.read_value_ptr<double2>(left_size);

  auto right_size = left_iter.read_meta<int>();
  auto right_points = left_iter.read_value_ptr<double2>(right_size);

  if (left_size == 0) {
    if (right_size == 0) {
      return Matrix("FFFFFFFF*");
    } else {
      return Matrix("FFFFFF01*");
    }
  }
  if(right_size == 0) {
    return Matrix("FF0FF1FF");
  }


  return de9im::INVALID_MATRIX;
}

// ops:
__device__ Matrix PointRelateOp(ConstIter& left_iter, WkbTag right_tag,
                                ConstIter& right_iter, Matrix matrix_hint) {
  (void)matrix_hint;  // ignore
  //  auto right_tag = right.get_tag(index);
  assert(right_tag.get_space_type() == WkbSpaceType::XY);
  auto left_point = left_iter.read_value<double2>();

  //  auto right_iter = right.get_iter(index);
  Matrix result;
  switch (right_tag.get_category()) {
    case WkbCategory::kPoint: {
      auto right_point = right_iter.read_value<double2>();
      auto is_eq = IsEqual(left_point, right_point);
      result = is_eq ? Matrix("0FFFFFFF*") : Matrix("FF0FFF0F*");
      break;
    }
    case WkbCategory::kLineString: {
      auto right_size = right_iter.read_meta<int>();
      auto right_points = right_iter.read_value_ptr<double2>(right_size);
      result = PointRelateToLineString(left_point, right_size, right_points);
      break;
    }
    default: {
      assert(false);
      result = de9im::INVALID_MATRIX;
      break;
    }
  }
  return result;
}

__device__ Matrix LineStringRelateOp(ConstIter& left_iter, WkbTag right_tag,
                                     ConstIter& right_iter, Matrix matrix_hint) {
  (void)matrix_hint;  // ignore
                      //  auto right_tag = right.get_tag(index);
  assert(right_tag.get_space_type() == WkbSpaceType::XY);

  auto left_size = left_iter.read_meta<int>();
  auto left_points = left_iter.read_value_ptr<double2>(left_size);
  Matrix result;
  switch (right_tag.get_category()) {
    case WkbCategory::kPoint: {
      auto right_point = right_iter.read_value<double2>();
      auto mat = PointRelateToLineString(right_point, left_size, left_points);
      result = mat.transpose();
      break;
    }
    case WkbCategory::kLineString: {
      auto right_size = right_iter.read_value<int>();
      auto right_points = right_iter.read_value_ptr<double2>(right_size);

      break;
    }
    default: {
      assert(false);
      result = de9im::INVALID_MATRIX;
      break;
    }
  }
  return result;
}

__device__ Matrix RelateOp(ConstGpuContext& left, ConstGpuContext& right,
                           de9im::Matrix matrix_hint, int index) {
  auto left_tag = left.get_tag(index);
  assert(left_tag.get_space_type() == WkbSpaceType::XY);
  de9im::Matrix result;
  auto left_iter = left.get_iter(index);
  auto right_iter = right.get_iter(index);

  switch (left_tag.get_category()) {
    case WkbCategory::kPoint: {
      result = PointRelateOp(left_iter, right.get_tag(index), right_iter, matrix_hint);
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
  assert(left_iter.values == left.get_value_ptr(index));
  assert(left_iter.metas == left.get_meta_ptr(index));
  return result;
}

static __global__ void ST_RelateImpl(ConstGpuContext left, ConstGpuContext right,
                                     de9im::Matrix input_matrix,
                                     de9im::Matrix* output_matrixes) {
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
