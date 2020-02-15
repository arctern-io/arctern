#pragma once

#include <memory>
#include <stdint.h>

#include "arrow/api.h"
#include "arrow/array.h"


namespace zilliz {
namespace gis {

//TODO: add description for each api

/**************************** GEOMETRY CONSTRUCTOR ***************************/

std::shared_ptr<arrow::Array>
ST_Point(const std::shared_ptr<arrow::Array> &x_values, 
         const std::shared_ptr<arrow::Array> &y_values);

std::shared_ptr<arrow::Array>
ST_PolygonFromEnvelope(const std::shared_ptr<arrow::Array> &min_x_values,
                       const std::shared_ptr<arrow::Array> &min_y_values,
                       const std::shared_ptr<arrow::Array> &max_x_values,
                       const std::shared_ptr<arrow::Array> &max_y_values);

/***************************** GEOMETRY ACCESSOR *****************************/

std::shared_ptr<arrow::Array>
ST_IsValid(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_IsSimple(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_GeometryType(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_NPoints(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_Envelope(const std::shared_ptr<arrow::Array> &geometries);


/**************************** GEOMETRY PROCESSING ****************************/

std::shared_ptr<arrow::Array>
ST_Buffer(const std::shared_ptr<arrow::Array> &geometries, 
          double buffer_distance, int n_quadrant_segments = 30);

// std::shared_ptr<arrow::Array>
// ST_PrecisionReduce(const std::shared_ptr<arrow::Array> &geometries, 
//                    int32_t precision);

std::shared_ptr<arrow::Array>
ST_Intersection(const std::shared_ptr<arrow::Array> &geometries_1,
                const std::shared_ptr<arrow::Array> &geometries_2);

std::shared_ptr<arrow::Array>
ST_MakeValid(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_SimplifyPreserveTopology(const std::shared_ptr<arrow::Array> &geometries, 
                            double distance_tolerance);

std::shared_ptr<arrow::Array>
ST_Centroid(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_ConvexHull(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_Transform(const std::shared_ptr<arrow::Array> &geos,
             const std::string &src_rs,
             const std::string &dst_rs);


/*************************** MEASUREMENT FUNCTIONS ***************************/

std::shared_ptr<arrow::Array>
ST_Distance(const std::shared_ptr<arrow::Array> &geometries_1,
            const std::shared_ptr<arrow::Array> &geometries_2);

std::shared_ptr<arrow::Array>
ST_Area(const std::shared_ptr<arrow::Array> &geometries);

std::shared_ptr<arrow::Array>
ST_Length(const std::shared_ptr<arrow::Array> &geometries);


/**************************** SPATIAL RELATIONSHIP ***************************/

std::shared_ptr<arrow::Array>
ST_Equals(const std::shared_ptr<arrow::Array> &geometries_1, 
          const std::shared_ptr<arrow::Array> &geometries_2);

std::shared_ptr<arrow::Array>
ST_Touches(const std::shared_ptr<arrow::Array> &geometries_1, 
           const std::shared_ptr<arrow::Array> &geometries_2);

std::shared_ptr<arrow::Array>
ST_Overlaps(const std::shared_ptr<arrow::Array> &geometries_1,
            const std::shared_ptr<arrow::Array> &geometries_2);

std::shared_ptr<arrow::Array>
ST_Crosses(const std::shared_ptr<arrow::Array> &geometries_1, 
           const std::shared_ptr<arrow::Array> &geometries_2);

std::shared_ptr<arrow::Array>
ST_Contains(const std::shared_ptr<arrow::Array> &geometries_1,
            const std::shared_ptr<arrow::Array> &geometries_2);

std::shared_ptr<arrow::Array>
ST_Intersects(const std::shared_ptr<arrow::Array> &geometries_1,
              const std::shared_ptr<arrow::Array> &geometries_2);

std::shared_ptr<arrow::Array>
ST_Within(const std::shared_ptr<arrow::Array> &geometries_1,
          const std::shared_ptr<arrow::Array> &geometries_2);


/*************************** AGGREGATE FUNCTIONS ***************************/

std::shared_ptr<arrow::Array>
ST_Union_Aggr(const std::shared_ptr<arrow::Array> &geometries);


std::shared_ptr<arrow::Array>
ST_Envelope_Aggr(const std::shared_ptr<arrow::Array> &geometries);


std::shared_ptr<std::string>
GIS_Version();


} // gis
} // zilliz
