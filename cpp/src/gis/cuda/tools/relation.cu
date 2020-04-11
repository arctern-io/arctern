#include "gis/cuda/tools/relation.h"

namespace arctern {
namespace gis {
namespace cuda {
using de9im::Matrix;
DEVICE_RUNNABLE bool PointOnInnerLineString(double2 left_point, int right_size,
                                            const double2* right_points) {
  // endpoints not included
  bool point_in_line = false;
  for (int i = 0; i < right_size - 1; ++i) {
    auto u = right_points[i];
    auto v = right_points[i + 1];
    if (i != 0 && IsEqual(u, left_point)) {
      point_in_line = true;
      break;
    }
    if (IsPointInLine(left_point, u, v)) {
      point_in_line = true;
      break;
    }
  }
  return point_in_line;
}

DEVICE_RUNNABLE bool LineStringOnLineString(int left_size, const double2* left_points,
                                            int right_size, const double2* right_points) {
  //
}

static constexpr double epsilon = 1e-8;
DEVICE_RUNNABLE inline bool is_zero(double x) { return abs(x) < epsilon; }

struct LineRelationResult {
  int II;             // dimension of II
  bool is_coveredby;  // is line coveredby linestring
  bool ep0_online;
  bool ep1_online;
};

DEVICE_RUNNABLE inline LineRelationResult LineOnLineString(const double2* line_endpoints,
                                                           int right_size,
                                                           const double2* right_points) {
  auto lv0 = to_complex(line_endpoints[0]);
  auto lv1 = to_complex(line_endpoints[0 + 1]) - lv0;
  LineRelationResult result{-1, false, false, false};
  for (int right_index = 0; right_index < right_size - 1; ++right_index) {
    auto rv0 = (to_complex(right_points[right_index]) - lv0) / lv1;
    auto rv1 = (to_complex(right_points[right_index + 1]) - lv0) / lv1;
    if (right_index != 0) {
      if (rv1 == 0) {
        result.ep0_online = true;
      }
      if (rv1 == 1) {
        result.ep1_online = true;
      }
    }
    if (is_zero(rv0.imag()) && is_zero(rv1.imag())) {
      auto r0 = rv0.real();
      auto r1 = rv0.real();
      if (r0 <= 0 && r1 <= 0 || r0 >= 1 && r1 >= 1) {
        // NOTE: ep0 at points false negative
        continue;
      }
      // intersect
      result.II = 1;
      if (0 <= r0 && r0 <= 1 && 0 <= r1 && r1 <= 1) {
        result.is_coveredby = true;
        break;
      }
    } else if (rv0.imag() * rv1.imag() <= 0) {
      auto proj =
          (rv0.real() * rv1.imag() - rv0.real() * rv0.imag()) / (rv1.imag() - rv0.imag());

      if (rv0.imag() == 0 && right_index == 0 ||
          rv1.imag() == 0 && right_index == right_size - 2) {
        continue;
      }

      if (0 < proj && proj < 1) {
        result.II = max(result.II, 0);
      }

      if (proj == 0) {
        result.ep0_online = true;
      }

      if (proj == 1) {
        result.ep1_online = true;
      }
    }
  }
  return {result.II, result.is_coveredby};
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
