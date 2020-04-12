#pragma once
#include <thrust/pair.h>

#include "gis/cuda/container/kernel_vector.h"
#include "gis/cuda/tools/de9im_matrix.h"
#include "gis/cuda/tools/relation.h"
namespace arctern {
namespace gis {
namespace cuda {
struct LineRelationResult {
  int II;             // dimension of II
  bool is_coveredby;  // is line coveredby LineString
  int cross_count;
};

struct KernelBuffer {
  KernelVector<thrust::pair<double, double>> ranges;
};

// endpoints included
DEVICE_RUNNABLE bool IsPointInLine(double2 point_raw, double2 line_beg, double2 line_end);
// return count of cross point
DEVICE_RUNNABLE int PointOnLineString(double2 left_point, int right_size,
                                      const double2* right_points);

// Note: when dealing with linestring, we view it as endpoints included
// linestring, which is collection of endpoints
// Known bug: false negative for
//  ST_Equals("LineString(0 0, 0 1, 0 2)", "LineString(0 0, 0 2)"));
// Solution was put off to next iteration
DEVICE_RUNNABLE LineRelationResult LineOnLineString(const double2* line_endpoints,
                                                    int right_size,
                                                    const double2* right_points,
                                                    KernelBuffer& buffer);

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
#include "gis/cuda/tools/relation.impl.h"
