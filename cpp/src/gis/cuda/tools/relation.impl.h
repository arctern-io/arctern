// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once
#include <thrust/complex.h>
#include <thrust/extrema.h>

#include <algorithm>
#include <utility>

#include "gis/cuda/container/kernel_vector.h"
#include "gis/cuda/tools/relation.h"

namespace arctern {
namespace gis {
namespace cuda {
using de9im::Matrix;

// TODO(dog): use global percision mode
DEVICE_RUNNABLE inline bool IsEqual(double2 a, double2 b) {
  return abs(a.x - b.x) == 0 && abs(a.y - b.y) == 0;
}

DEVICE_RUNNABLE inline auto to_complex(double2 point) {
  return thrust::complex<double>(point.x, point.y);
}

// endpoints included
DEVICE_RUNNABLE inline bool IsPointInLine(double2 point_raw, double2 line_beg,
                                          double2 line_end) {
  thrust::complex<double> point = to_complex(point_raw);
  auto v0 = point - to_complex(line_beg);
  auto v1 = point - to_complex(line_end);
  auto result = v0 * thrust::conj(v1);

  return result.imag() == 0 && result.real() <= 0;
}

// return count of cross point
DEVICE_RUNNABLE inline int PointOnLineString(double2 left_point, int right_size,
                                             const double2* right_points) {
  int count = 0;
  for (int i = 0; i < right_size - 1; ++i) {
    auto u = right_points[i];
    auto v = right_points[i + 1];
    if (IsPointInLine(left_point, u, v)) {
      ++count;
    }
  }
  return count;
}

DEVICE_RUNNABLE inline bool is_zero(double x) {
  //  static constexpr double epsilon = 1e-8;
  //  (void)epsilon;
  return x == 0;
}

// check if [0, 1] coveredby sorted ranges
DEVICE_RUNNABLE inline bool IsRange01CoveredBy(
    const KernelVector<thrust::pair<double, double>>& ranges) {
  auto total_range = ranges[0];
  for (int iter = 1; iter < ranges.size(); ++iter) {
    auto range = ranges[iter];
    if (total_range.second < range.first) {
      // has gap
      if (range.first <= 0) {
        // previous total range are all before 0. just discard
        total_range = range;
      } else {
        // gap after 0. terminate
        break;
      }
    } else {
      // no gap, just extend
      total_range.second = thrust::max(total_range.second, range.second);
      if (range.second >= 1) {
        break;
      }
    }
  }
  return total_range.first <= 0 && 1 <= total_range.second;
}

// Note: when dealing with linestring, we view it as endpoints included
// linestring, which is collection of endpoints
DEVICE_RUNNABLE inline LineRelationResult LineOnLineString(const double2* line_endpoints,
                                                           int right_size,
                                                           const double2* right_points,
                                                           KernelBuffer& buffer) {
  // possible false negative, record for further processing
  auto& ranges_buffer = buffer.ranges;
  ranges_buffer.clear();
  // this is to avoid too many allocations
  bool has_first_item = false;
  thrust::pair<double, double> first_item;

  // project left line to 0->1
  auto lv0 = to_complex(line_endpoints[0]);
  auto lv1 = to_complex(line_endpoints[0 + 1]) - lv0;
  LineRelationResult result{-1, false, 0};
  for (int right_index = 0; right_index < right_size - 1; ++right_index) {
    // do similiar projection
    auto rv0 = (to_complex(right_points[right_index]) - lv0) / lv1;
    auto rv1 = (to_complex(right_points[right_index + 1]) - lv0) / lv1;
    // if projected complex nums are at x axis
    if (is_zero(rv0.imag()) && is_zero(rv1.imag())) {
      auto r0 = rv0.real();
      auto r1 = rv1.real();
      if ((r0 <= 0 && r1 <= 0) || (r0 >= 1 && r1 >= 1)) {
        // outside, just check endpoints
        if (r0 == 0 || r0 == 1) {
          ++result.cross_count;
          result.CC = thrust::max(result.CC, 0);
        } else if (r1 == 0 || r1 == 1) {
          ++result.cross_count;
          result.CC = thrust::max(result.CC, 0);
        }
      } else {
        // at least intersect
        auto rmin = thrust::min(r0, r1);
        auto rmax = thrust::max(r0, r1);
        result.CC = 1;
        // check if overlap
        if (rmin <= 0 && 1 <= rmax) {
          result.is_coveredby = true;
          break;
        } else {
          // false negative since multiple segments can be combined
          if (!result.is_coveredby) {
            auto item = thrust::make_pair(rmin, rmax);
            // first_item lazy push, so common cases won't cost memory allocation
            if (!has_first_item) {
              has_first_item = true;
              first_item = item;
            } else {
              ranges_buffer.push_back(item);
            }
          }
        }
      }
    } else if (rv0.imag() * rv1.imag() <= 0) {
      // cross/touch the x axis, so check intersect point
      auto proj =
          (rv0.real() * rv1.imag() - rv1.real() * rv0.imag()) / (rv1.imag() - rv0.imag());
      if (0 <= proj && proj <= 1) {
        result.CC = thrust::max(result.CC, 0);
        ++result.cross_count;
      }
    }
  }

  // if vector has value
  if (!result.is_coveredby && ranges_buffer.size() != 0) {
    assert(has_first_item);
    ranges_buffer.push_back(first_item);
    ranges_buffer.sort();
    result.is_coveredby = IsRange01CoveredBy(ranges_buffer);
  }

  return result;
}

// return sum result of lineOnlineString
DEVICE_RUNNABLE inline LineRelationResult SumLineOnLineString(int left_size,
                                                              const double2* left_points,
                                                              int right_size,
                                                              const double2* right_points,
                                                              KernelBuffer& buffer) {
  assert(left_size >= 2);
  assert(right_size >= 2);
  LineRelationResult total_relation{-1, true, 0};
  for (auto index = 0; index < left_size - 1; ++index) {
    auto left_ptr = reinterpret_cast<const double2*>(left_points + index);
    auto relation = LineOnLineString(left_ptr, right_size, right_points, buffer);
    total_relation.CC = thrust::max(total_relation.CC, relation.CC);
    total_relation.is_coveredby = total_relation.is_coveredby && relation.is_coveredby;
    total_relation.cross_count = total_relation.cross_count + relation.cross_count;
  }
  return total_relation;
}

DEVICE_RUNNABLE inline Matrix LineStringRelateToLineString(int left_size,
                                                           const double2* left_points,
                                                           int right_size,
                                                           const double2* right_points,
                                                           KernelBuffer& buffer) {
  if (left_size == 0) {
    if (right_size == 0) {
      return Matrix("FFFFFFFF*");
    } else {
      return Matrix("FFFFFF10*");
    }
  }
  if (right_size == 0) {
    return Matrix("FF1FF0FF*");
  }
  assert(left_size >= 2);
  assert(right_size >= 2);
  // left boundary
  auto left_b = thrust::make_pair(left_points[0], left_points[left_size - 1]);
  auto right_b = thrust::make_pair(right_points[0], right_points[right_size - 1]);
  // boundary to boundary
  int BB_count = 0;
  BB_count += IsEqual(left_b.first, right_b.first);
  BB_count += IsEqual(left_b.first, right_b.second);
  BB_count += IsEqual(left_b.second, right_b.first);
  BB_count += IsEqual(left_b.second, right_b.second);
  // boundary to linestring
  auto BC_count_0 = PointOnLineString(left_b.first, right_size, right_points);
  auto BC_count_1 = PointOnLineString(left_b.second, right_size, right_points);
  auto CB_count_0 = PointOnLineString(right_b.first, left_size, left_points);
  auto CB_count_1 = PointOnLineString(right_b.second, left_size, left_points);
  auto BC_count = BC_count_0 + BC_count_1;
  auto CB_count = CB_count_0 + CB_count_1;

  auto IE_relation =
      SumLineOnLineString(left_size, left_points, right_size, right_points, buffer);

  auto EI_relation =
      SumLineOnLineString(right_size, right_points, left_size, left_points, buffer);
  assert(IE_relation.CC == EI_relation.CC);
  assert(IE_relation.CC == 1 || IE_relation.cross_count == EI_relation.cross_count);
  using State = de9im::Matrix::State;
  Matrix matrix;
  switch (IE_relation.CC) {
    case -1: {
      matrix->II = State::kFalse;
      break;
    }
    case 0: {
      auto II_count = IE_relation.cross_count - BC_count - CB_count + BB_count;
      matrix->II = II_count ? State::kDim0 : State::kFalse;
      break;
    }
    case 1: {
      matrix->II = State::kDim1;
      break;
    }
    default: {
      matrix->II = State::kInvalid;
      assert(false);
    }
  }
  matrix->BI = BC_count - BB_count ? State::kDim0 : State::kFalse;
  matrix->IB = CB_count - BB_count ? State::kDim0 : State::kFalse;
  matrix->IE = !IE_relation.is_coveredby ? State::kDim1 : State::kFalse;
  matrix->EI = !EI_relation.is_coveredby ? State::kDim1 : State::kFalse;
  // Fix bug: BC_count != 2 can be misleading if B is on vertex of I,
  // which will be counted twice
  matrix->BE = !(BC_count_0 && BC_count_1) ? State::kDim0 : State::kFalse;
  matrix->EB = !(CB_count_0 && CB_count_1) ? State::kDim0 : State::kFalse;
  matrix->BB = BB_count ? State::kDim0 : State::kFalse;

  return matrix;
}

DEVICE_RUNNABLE Matrix PointRelateToLineString(double2 left_point, int right_size,
                                               const double2* right_points) {
  if (right_size == 0) {
    return Matrix("FFFFFFFF*");
  }

  if (right_size == 1) {
    //    auto right_point = right_points[0];
    //    auto is_eq = IsEqual(left_point, right_point);
    //    return is_eq ? Matrix("F0FFFFF0*") : Matrix("FF0FFFF0*");
    return de9im::INVALID_MATRIX;
  }

  assert(right_size >= 2);
  Matrix mat;

  using State = Matrix::State;
  using Position = Matrix::Position;

  auto C_count = PointOnLineString(left_point, right_size, right_points);

  // endpoints
  auto ep0 = right_points[0];
  auto ep1 = right_points[right_size - 1];
  int B_count = (int)IsEqual(left_point, ep0) + (int)IsEqual(left_point, ep1);

  assert(C_count - B_count >= 0);
  mat->II = C_count - B_count ? State::kDim0 : State::kFalse;
  mat->IB = B_count ? State::kDim0 : State::kFalse;
  mat->IE = !C_count ? State::kDim0 : State::kFalse;
  mat.set_row<Position::kB>("FFF");
  mat->EI = State::kDim1;
  mat->EB = B_count != 2 ? State::kDim0 : State::kFalse;

  return mat;
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
