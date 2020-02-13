#ifndef GIS_H
#define GIS_H

#include "arrow/api.h"

std::shared_ptr<arrow::Array>
point_map(std::shared_ptr<arrow::Array> arr_x,std::shared_ptr<arrow::Array> arr_y);

std::shared_ptr<arrow::Array>
make_point(std::shared_ptr<arrow::Array> arr_x,std::shared_ptr<arrow::Array> arr_y);

namespace zilliz {
namespace gis {

std::shared_ptr<arrow::Array>
ST_Point(const std::shared_ptr<arrow::Array> &point_x, const std::shared_ptr<arrow::Array> &point_y);


std::shared_ptr<arrow::Array>
ST_Intersection(const std::shared_ptr<arrow::Array> &left_geometries,
                const std::shared_ptr<arrow::Array> &right_geometries);


std::shared_ptr<arrow::Array>
ST_IsValid(const std::shared_ptr<arrow::Array> &geometries);


std::shared_ptr<arrow::Array>
ST_Equals(const std::shared_ptr<arrow::Array> &left_geometries, const std::shared_ptr<arrow::Array> &right_geometries);


std::shared_ptr<arrow::Array>
ST_Touches(const std::shared_ptr<arrow::Array> &left_geometries, const std::shared_ptr<arrow::Array> &right_geometries);


std::shared_ptr<arrow::Array>
ST_Overlaps(const std::shared_ptr<arrow::Array> &left_geometries,
            const std::shared_ptr<arrow::Array> &right_geometries);


std::shared_ptr<arrow::Array>
ST_Crosses(const std::shared_ptr<arrow::Array> &left_geometries, const std::shared_ptr<arrow::Array> &right_geometries);


std::shared_ptr<arrow::Array>
ST_IsSimple(const std::shared_ptr<arrow::Array> &geometries);


// std::shared_ptr<arrow::Array>
// ST_PrecisionReduce(const std::shared_ptr<arrow::Array> &geometries, int32_t num_dot);


std::shared_ptr<arrow::Array>
ST_GeometryType(const std::shared_ptr<arrow::Array> &geometries);


std::shared_ptr<arrow::Array>
ST_MakeValid(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_SimplifyPreserveTopology(const std::shared_ptr<arrow::Array> &geometries, double distanceTolerance);


std::shared_ptr<arrow::Array>
ST_PolygonFromEnvelope(const std::shared_ptr<arrow::Array> &min_x,
                       const std::shared_ptr<arrow::Array> &min_y,
                       const std::shared_ptr<arrow::Array> &max_x,
                       const std::shared_ptr<arrow::Array> &max_y);

std::shared_ptr<arrow::Array>
ST_Contains(const std::shared_ptr<arrow::Array> &ptr_x,
            const std::shared_ptr<arrow::Array> &ptr_y);

std::shared_ptr<arrow::Array>
ST_Intersects(const std::shared_ptr<arrow::Array> &geo_arr1,
              const std::shared_ptr<arrow::Array> &geo_arr2);

std::shared_ptr<arrow::Array>
ST_Within(const std::shared_ptr<arrow::Array> &geo_arr1,
          const std::shared_ptr<arrow::Array> &geo_arr2);

std::shared_ptr<arrow::Array>
ST_Distance(const std::shared_ptr<arrow::Array> &geo_arr1,
            const std::shared_ptr<arrow::Array> &geo_arr2);

std::shared_ptr<arrow::Array>
ST_Area(const std::shared_ptr<arrow::Array> &geo_arr);

std::shared_ptr<arrow::Array>
ST_Centroid(const std::shared_ptr<arrow::Array> &geo_arr);

std::shared_ptr<arrow::Array>
ST_Length(const std::shared_ptr<arrow::Array> &geo_arr);

std::shared_ptr<arrow::Array>
ST_ConvexHull(const std::shared_ptr<arrow::Array> &geo_arr);

std::shared_ptr<arrow::Array>
ST_NPoints(const std::shared_ptr<arrow::Array> &geo_arr);

std::shared_ptr<arrow::Array>
ST_Envelope(const std::shared_ptr<arrow::Array> &geo_arr);

std::shared_ptr<arrow::Array>
ST_Buffer(const std::shared_ptr<arrow::Array> &geometries, 
          double buffer_distance, int n_quadrant_segments = 30);

} // gis
} // zilliz

#endif
