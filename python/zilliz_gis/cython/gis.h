
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


std::shared_ptr<arrow::Array>
ST_PrecisionReduce(const std::shared_ptr<arrow::Array> &geometries, int32_t num_dot);


std::shared_ptr<arrow::Array>
ST_GeometryType(const std::shared_ptr<arrow::Array> &geometries);


std::shared_ptr<arrow::Array>
ST_MakeValid(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_SimplifyPreserveTopology(const std::shared_ptr<arrow::Array> &geometries, double distanceTolerance);
}
}


#endif //GIS_H
