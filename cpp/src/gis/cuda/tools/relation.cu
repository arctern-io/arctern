#include "gis/cuda/tools/relation.h"

namespace arctern {
namespace gis {
namespace cuda {
using de9im::Matrix;

// TODO(dog): use global percision mode
DEVICE_RUNNABLE bool IsEqual(double2 a, double2 b) {
  return abs(a.x - b.x) == 0 && abs(a.y - b.y) == 0;
}

DEVICE_RUNNABLE auto to_complex(double2 point) {
  return thrust::complex<double>(point.x, point.y);
}

// endpoints included
DEVICE_RUNNABLE bool IsPointInLine(double2 point_raw, double2 line_beg,
                                   double2 line_end) {
  thrust::complex<double> point = to_complex(point_raw);
  auto v0 = point - to_complex(line_beg);
  auto v1 = point - to_complex(line_end);
  auto result = v0 * thrust::conj(v1);

  return result.imag() == 0 && result.real() <= 0;
}

// return count of cross point
DEVICE_RUNNABLE int PointOnInnerLineString(double2 left_point, int right_size,
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

// Note: when dealing with linestring, we view it as endpoints included
// linestring, which is collection of endpoints
// Known bug: false negative for
//  ST_Equals("LineString(0 0, 0 1, 0 2)", "LineString(0 0, 0 2)"));
// Solution was put off to next iteration
DEVICE_RUNNABLE LineRelationResult LineOnLineString(const double2* line_endpoints,
                                                    int right_size,
                                                    const double2* right_points) {
  // included
  auto lv0 = to_complex(line_endpoints[0]);
  // not included
  auto lv1 = to_complex(line_endpoints[0 + 1]) - lv0;
  LineRelationResult result{-1, false, 0};
  for (int right_index = 0; right_index < right_size - 1; ++right_index) {
    // included
    auto rv0 = (to_complex(right_points[right_index]) - lv0) / lv1;
    // not included
    auto rv1 = (to_complex(right_points[right_index + 1]) - lv0) / lv1;
    if (is_zero(rv0.imag()) && is_zero(rv1.imag())) {
      // included
      auto r0 = rv0.real();
      // not included
      auto r1 = rv0.real();
      if (r0 <= 0 && r1 <= 0 || r0 >= 1 && r1 >= 1) {
        if (r0 == 0 || r0 == 1) {
          ++result.cross_count;
          result.II = std::max(result.II, 0);
        }
        if (r1 == 0 || r1 == 1) {
          ++result.cross_count;
          result.II = std::max(result.II, 0);
        }
      } else {
        // at least intersect, no need for cross_count
        result.II = 1;
        if (0 <= r0 && r0 <= 1 && 0 <= r1 && r1 <= 1) {
          result.is_coveredby = true;
          break;
        } else {
          // possible false negative, record for further processing
          result.overlap_count++;

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
  return result;
}

DEVICE_RUNNABLE bool LineStringOnLineString(int left_size, const double2* left_points,
                                            int right_size, const double2* right_points) {

}

DEVICE_RUNNABLE de9im::Matrix LineStringRelateToLineString(int left_size,
                                                           const double2* left_points,
                                                           int right_size,
                                                           const double2* right_points) {
  assert(left_size >= 2);
  assert(right_size >= 2);

  for (int left_index = 0; left_index < left_size - 1; ++left_index) {
    // NOTE: we assume SIMPLE geometries, so LineString(0 0, 2 0, 1 0)
    // won't be handled correctly since it is not simple
    // Before use this method, geometries must be simplify

    auto lv0 = to_complex(left_points[left_index]);
    auto lv1 = to_complex(left_points[left_index + 1]) - lv0;

    for (int right_index = 0; right_index < right_size - 1; ++left_index) {
      // project points to zeros

      auto rv0 = (to_complex(right_points[right_index]) - lv0) / lv1;
      auto rv1 = (to_complex(right_points[right_index + 1]) - lv0) / lv1;
      Matrix sub_mat("FFFFFFFF*");
      int II = -1, IE = 1, EI = 1;
      if (is_zero(rv0.imag()) && is_zero(rv1.imag())) {
        double lv0_ref = 0;
        double lv1_ref = 1;
        // maybe cross
        auto min_ep = min(rv0.real(), rv1.real());
        auto max_ep = max(rv0.real(), rv1.real());
        if (max_ep <= 0 || min_ep >= 1) {
          continue;
        }
        II = 1;
        if (max_ep <= 1 || min_ep >= 0) {
          IE = -1;
        }
      }
    }
  }
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
