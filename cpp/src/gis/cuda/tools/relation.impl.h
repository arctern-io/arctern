#pragma once
#include <thrust/complex.h>
#include <thrust/optional.h>

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
      total_range.second = max(total_range.second, range.second);
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
  thrust::optional<thrust::pair<double, double>> first_item;

  auto lv0 = to_complex(line_endpoints[0]);
  auto lv1 = to_complex(line_endpoints[0 + 1]) - lv0;
  LineRelationResult result{-1, false, 0};
  for (int right_index = 0; right_index < right_size - 1; ++right_index) {
    auto rv0 = (to_complex(right_points[right_index]) - lv0) / lv1;
    auto rv1 = (to_complex(right_points[right_index + 1]) - lv0) / lv1;
    if (is_zero(rv0.imag()) && is_zero(rv1.imag())) {
      // included
      auto r0 = rv0.real();
      // not included
      auto r1 = rv0.real();
      if ((r0 <= 0 && r1 <= 0) || (r0 >= 1 && r1 >= 1)) {
        if (r0 == 0 || r0 == 1) {
          ++result.cross_count;
          result.II = max(result.II, 0);
        }
        if (r1 == 0 || r1 == 1) {
          ++result.cross_count;
          result.II = max(result.II, 0);
        }
      } else {
        // at least intersect, no need for cross_count
        result.II = 1;
        if (0 <= r0 && r0 <= 1 && 0 <= r1 && r1 <= 1) {
          result.is_coveredby = true;
          break;
        } else {
          if (!result.is_coveredby) {
            auto item = thrust::make_pair(min(r0, r1), max(r0, r1));
            if (!first_item.has_value()) {
              first_item = item;
            } else {
              ranges_buffer.push_back(item);
            }
          }
        }
      }
    } else if (rv0.imag() * rv1.imag() <= 0) {
      auto proj =
          (rv0.real() * rv1.imag() - rv0.real() * rv0.imag()) / (rv1.imag() - rv0.imag());
      if (0 <= proj && proj <= 1) {
        result.II = max(result.II, 0);
        ++result.cross_count;
      }
    }
  }

  if (!result.is_coveredby && ranges_buffer.size() != 0) {
    assert(first_item.has_value());
    ranges_buffer.push_back(first_item.value_or(thrust::make_pair(0.0, 0.0)));
    ranges_buffer.sort();
    result.is_coveredby = IsRange01CoveredBy(ranges_buffer);
  }

  return result;
}

DEVICE_RUNNABLE inline de9im::Matrix LineStringRelateToLineString(
    int left_size, const double2* left_points, int right_size,
    const double2* right_points) {
  assert(left_size >= 2);
  assert(right_size >= 2);
  return de9im::INVALID_MATRIX;
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
