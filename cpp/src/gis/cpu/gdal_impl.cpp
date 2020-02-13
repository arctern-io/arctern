
#include "gis/api.h"
#include "utils/check_status.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ogr_api.h>
#include <ogrsf_frmts.h>


namespace zilliz {
namespace gis {


#define UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T1(\
RESULT_BUILDER_TYPE, GEO_VAR, APPEND_RESULT)   \
do {    \
    auto len = geometries->length();    \
    auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);    \
    RESULT_BUILDER_TYPE builder;   \
    void *GEO_VAR;    \
    for (int32_t i = 0; i < len; i++) { \
        CHECK_GDAL(OGRGeometryFactory::createFromWkt(\
            wkt_geometries->GetString(i).c_str(), nullptr, (OGRGeometry**)(&GEO_VAR))); \
        CHECK_ARROW(builder.Append(APPEND_RESULT));  \
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR);   \
    }   \
    std::shared_ptr<arrow::Array> results;  \
    CHECK_ARROW(builder.Finish(&results));   \
    return results; \
} while(0)


#define UNARY_WKT_FUNC_WITH_GDAL_IMPL_T1(\
FUNC_NAME, RESULT_BUILDER_TYPE, GEO_VAR, APPEND_RESULT)   \
std::shared_ptr<arrow::Array> \
FUNC_NAME(const std::shared_ptr<arrow::Array> &geometries) {   \
    UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T1(RESULT_BUILDER_TYPE, GEO_VAR, APPEND_RESULT);   \
}


#define UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(\
RESULT_BUILDER_TYPE, GEO_VAR, APPEND_RESULT)   \
do {    \
    auto len = geometries->length();    \
    auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);    \
    RESULT_BUILDER_TYPE builder;   \
    void *GEO_VAR, *geo_tmp;    \
    char *wkt_tmp;    \
    for (int32_t i = 0; i < len; i++) { \
        CHECK_GDAL(OGRGeometryFactory::createFromWkt(   \
            wkt_geometries->GetString(i).c_str(), nullptr, (OGRGeometry**)(&GEO_VAR))); \
        geo_tmp = APPEND_RESULT;	\
        CHECK_GDAL(OGR_G_ExportToWkt(geo_tmp, &wkt_tmp));  \
        CHECK_ARROW(builder.Append(wkt_tmp)); \
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR);   \
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo_tmp);   \
        CPLFree(wkt_tmp); \
    }   \
    std::shared_ptr<arrow::Array> results;  \
    CHECK_ARROW(builder.Finish(&results));   \
    return results; \
} while (0)


#define UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(\
FUNC_NAME, RESULT_BUILDER_TYPE, GEO_VAR, APPEND_RESULT)   \
std::shared_ptr<arrow::Array> \
FUNC_NAME(const std::shared_ptr<arrow::Array> &geometries) {   \
    UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(RESULT_BUILDER_TYPE, GEO_VAR, APPEND_RESULT); \
}


#define BINARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T1(\
RESULT_BUILDER_TYPE, GEO_VAR_1, GEO_VAR_2, APPEND_RESULT)   \
do {    \
    assert(geometries_1->length() == geometries_2->length());   \
    auto len = geometries_1->length();  \
    auto wkt_geometries_1 = std::static_pointer_cast<arrow::StringArray>(geometries_1); \
    auto wkt_geometries_2 = std::static_pointer_cast<arrow::StringArray>(geometries_2); \
    RESULT_BUILDER_TYPE builder;  \
    void *GEO_VAR_1, *GEO_VAR_2;   \
    for (int32_t i = 0; i < len; i++) { \
        CHECK_GDAL(OGRGeometryFactory::createFromWkt(   \
            wkt_geometries_1->GetString(i).c_str(), nullptr, (OGRGeometry**)(&GEO_VAR_1))); \
        CHECK_GDAL(OGRGeometryFactory::createFromWkt(   \
            wkt_geometries_2->GetString(i).c_str(), nullptr, (OGRGeometry**)(&GEO_VAR_2))); \
        CHECK_ARROW(builder.Append(APPEND_RESULT));  \
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR_1);   \
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR_2);   \
    }   \
    std::shared_ptr<arrow::Array> results;  \
    CHECK_ARROW(builder.Finish(&results));   \
    return results; \
} while(0)


#define UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T3(\
RESULT_BUILDER_TYPE, GEO_WKT_VAR, APPEND_RESULT)   \
do {    \
    auto len = geometries->length();    \
    auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);    \
    RESULT_BUILDER_TYPE builder;   \
    const char *GEO_WKT_VAR;    \
    for (int32_t i = 0; i < len; i++) { \
        auto geo_wkt_str = wkt_geometries->GetString(i);    \
        GEO_WKT_VAR = geo_wkt_str.c_str(); \
        CHECK_ARROW(builder.Append(APPEND_RESULT));  \
    }   \
    std::shared_ptr<arrow::Array> results;  \
    CHECK_ARROW(builder.Finish(&results));   \
    return results; \
} while(0)


#define UNARY_WKT_FUNC_WITH_GDAL_IMPL_T3(\
FUNC_NAME, RESULT_BUILDER_TYPE, GEO_WKT_VAR, APPEND_RESULT)   \
std::shared_ptr<arrow::Array> \
FUNC_NAME(const std::shared_ptr<arrow::Array> &geometries) {   \
    UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T3(RESULT_BUILDER_TYPE, GEO_WKT_VAR, APPEND_RESULT);   \
}


#define BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(\
FUNC_NAME, RESULT_BUILDER_TYPE, GEO_VAR_1, GEO_VAR_2, APPEND_RESULT)   \
std::shared_ptr<arrow::Array>   \
FUNC_NAME(const std::shared_ptr<arrow::Array> &geometries_1,    \
          const std::shared_ptr<arrow::Array> &geometries_2) {  \
    BINARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T1( \
        RESULT_BUILDER_TYPE, GEO_VAR_1, GEO_VAR_2, APPEND_RESULT);   \
}


#define BINARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(\
RESULT_BUILDER_TYPE, GEO_VAR_1, GEO_VAR_2, APPEND_RESULT)   \
do {    \
    assert(geometries_1->length() == geometries_2->length());   \
    auto len = geometries_1->length();  \
    auto wkt_geometries_1 = std::static_pointer_cast<arrow::StringArray>(geometries_1); \
    auto wkt_geometries_2 = std::static_pointer_cast<arrow::StringArray>(geometries_2); \
    RESULT_BUILDER_TYPE builder;  \
    void *GEO_VAR_1, *GEO_VAR_2, *geo_tmp;   \
    char *wkt_tmp; \
    for (int32_t i = 0; i < len; i++) { \
        CHECK_GDAL(OGRGeometryFactory::createFromWkt(   \
            wkt_geometries_1->GetString(i).c_str(), nullptr, (OGRGeometry**)(&GEO_VAR_1))); \
        CHECK_GDAL(OGRGeometryFactory::createFromWkt(   \
            wkt_geometries_2->GetString(i).c_str(), nullptr, (OGRGeometry**)(&GEO_VAR_2))); \
        geo_tmp = APPEND_RESULT;   \
        CHECK_GDAL(OGR_G_ExportToWkt(geo_tmp, &wkt_tmp));  \
        CHECK_ARROW(builder.Append(wkt_tmp));  \
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR_1);   \
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)GEO_VAR_2);   \
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo_tmp);   \
        CPLFree(wkt_tmp); \
    }   \
    std::shared_ptr<arrow::Array> results;  \
    CHECK_ARROW(builder.Finish(&results));   \
    return results; \
} while(0)


#define BINARY_WKT_FUNC_WITH_GDAL_IMPL_T2(\
FUNC_NAME, RESULT_BUILDER_TYPE, GEO_VAR_1, GEO_VAR_2, APPEND_RESULT)   \
std::shared_ptr<arrow::Array> \
FUNC_NAME(const std::shared_ptr<arrow::Array> &geometries_1,    \
          const std::shared_ptr<arrow::Array> &geometries_2) {  \
    BINARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2( \
        RESULT_BUILDER_TYPE, GEO_VAR_1, GEO_VAR_2, APPEND_RESULT);  \
}


inline void *
Wrapper_OGR_G_Centroid(void *geo) {
    void *centroid = new OGRPoint();
    OGR_G_Centroid(geo, centroid);
    return centroid;
}


inline bool
Wrapper_OGR_G_IsValid(const char *geo_wkt) {
    void *geo = nullptr;
    bool is_valid = false;
    auto err_code = OGRGeometryFactory::createFromWkt(geo_wkt, nullptr, (OGRGeometry**)(&geo));
    if (err_code == OGRERR_NONE) {
        is_valid = OGR_G_IsValid(geo) != 0;
        OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo);
    }
    return is_valid;
}


/************************ GEOMETRY CONSTRUCTOR ************************/

std::shared_ptr<arrow::Array>
ST_Point(const std::shared_ptr<arrow::Array> &x_values,
         const std::shared_ptr<arrow::Array> &y_values) {

    assert(x_values->length() == y_values->length());
    auto len = x_values->length();
    auto x_double_values = std::static_pointer_cast<arrow::DoubleArray>(x_values);
    auto y_double_values = std::static_pointer_cast<arrow::DoubleArray>(y_values);
    OGRPoint point;
    char *wkt = nullptr;
    arrow::StringBuilder builder;

    for (int32_t i = 0; i < len; i++) {
        point.setX(x_double_values->Value(i));
        point.setY(y_double_values->Value(i));
        CHECK_GDAL(OGR_G_ExportToWkt(&point, &wkt));
        CHECK_ARROW(builder.Append(wkt));
        CPLFree(wkt);
    }
    std::shared_ptr<arrow::Array> results;
    CHECK_ARROW(builder.Finish(&results));
    return results;
}

std::shared_ptr<arrow::Array>
ST_PolygonFromEnvelope(const std::shared_ptr<arrow::Array> &min_x_values,
                       const std::shared_ptr<arrow::Array> &min_y_values,
                       const std::shared_ptr<arrow::Array> &max_x_values,
                       const std::shared_ptr<arrow::Array> &max_y_values) {

    assert(min_x_values->length() == max_x_values->length());
    assert(min_y_values->length() == max_y_values->length());
    assert(min_x_values->length() == min_y_values->length());
    auto len = min_x_values->length();

    auto min_x_double_values = std::static_pointer_cast<const arrow::DoubleArray>(min_x_values);
    auto min_y_double_values = std::static_pointer_cast<const arrow::DoubleArray>(min_y_values);
    auto max_x_double_values = std::static_pointer_cast<const arrow::DoubleArray>(max_x_values);
    auto max_y_double_values = std::static_pointer_cast<const arrow::DoubleArray>(max_y_values);

    arrow::StringBuilder builder;

    for (int32_t i = 0; i < len; i++) {
        OGRLinearRing ring;
        ring.addPoint(min_x_double_values->Value(i), min_y_double_values->Value(i));
        ring.addPoint(max_x_double_values->Value(i), min_y_double_values->Value(i));
        ring.addPoint(min_x_double_values->Value(i), max_y_double_values->Value(i));
        ring.addPoint(max_x_double_values->Value(i), max_y_double_values->Value(i));
        ring.addPoint(min_x_double_values->Value(i), min_y_double_values->Value(i));
        ring.closeRings();
        OGRPolygon polygon;
        polygon.addRing(&ring);
        char *wkt = nullptr;
        CHECK_GDAL(OGR_G_ExportToWkt(&polygon, &wkt));
        CHECK_ARROW(builder.Append(wkt));
        CPLFree(wkt);
    }
    std::shared_ptr<arrow::Array> results;
    CHECK_ARROW(builder.Finish(&results));
    return results;
}


/************************* GEOMETRY ACCESSOR **************************/

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T3(
    ST_IsValid, arrow::BooleanBuilder, geo_wkt, 
    Wrapper_OGR_G_IsValid(geo_wkt));

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_IsSimple, arrow::BooleanBuilder, geo, 
    OGR_G_IsSimple(geo) != 0);

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_GeometryType, arrow::StringBuilder, geo, 
    OGR_G_GetGeometryName(geo));

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_NPoints, arrow::Int64Builder, geo, 
    OGR_G_GetPointCount(geo));

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(
    ST_Envelope, arrow::StringBuilder, geo,
    OGR_G_Boundary(geo));


/************************ GEOMETRY PROCESSING ************************/

std::shared_ptr<arrow::Array>
ST_Buffer(const std::shared_ptr<arrow::Array> &geometries, 
          double buffer_distance, int n_quadrant_segments) {
    UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(
        arrow::StringBuilder, geo, OGR_G_Buffer(geo, buffer_distance, n_quadrant_segments));
}

// std::shared_ptr<arrow::Array>
// ST_PrecisionReduce(const std::shared_ptr<arrow::Array> &geometries,
//                    int32_t precision) {

//     char precision_str[32];
//     sprintf(precision_str, "%i", precision);

//     const char *prev_config = CPLGetConfigOption("OGR_WKT_PRECISION", nullptr);
//     char *old_precision_str = prev_config ? CPLStrdup(prev_config) : nullptr;
//     CPLSetConfigOption("OGR_WKT_PRECISION", precision_str);

//     auto len = geometries->length();
//     auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);
//     arrow::StringBuilder builder;
//     void *geo;
//     char *wkt_tmp;
//     for (int32_t i = 0; i < len; i++) {
//         CHECK_GDAL(OGRGeometryFactory::createFromWkt(
//             wkt_geometries->GetString(i).c_str(), nullptr, (OGRGeometry**)(&geo)));
//         CHECK_GDAL(OGR_G_ExportToWkt(geo, &wkt_tmp));
//         CHECK_ARROW(builder.Append(wkt_tmp));
//         OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo);
//         CPLFree(wkt_tmp);
//     }

//     CPLSetConfigOption("OGR_WKT_PRECISION", old_precision_str);
//     CPLFree(old_precision_str);

//     std::shared_ptr<arrow::Array> results;
//     CHECK_ARROW(builder.Finish(&results));
//     return results; 
// }
 
BINARY_WKT_FUNC_WITH_GDAL_IMPL_T2(
    ST_Intersection, arrow::StringBuilder, geo_1, geo_2,
    OGR_G_Intersection(geo_1, geo_2));

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(
    ST_MakeValid, arrow::StringBuilder, geo,
    OGR_G_MakeValid(geo));

std::shared_ptr<arrow::Array>
ST_SimplifyPreserveTopology(const std::shared_ptr<arrow::Array> &geometries, 
                            double distance_tolerance) {
    UNARY_WKT_FUNC_BODY_WITH_GDAL_IMPL_T2(
        arrow::StringBuilder, geo, OGR_G_SimplifyPreserveTopology(geo, distance_tolerance));
}
 
UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(
    ST_Centroid, arrow::StringBuilder, geo,
    Wrapper_OGR_G_Centroid(geo));

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T2(
    ST_ConvexHull, arrow::StringBuilder, geo,
    OGR_G_ConvexHull(geo));


/************************ MEASUREMENT FUNCTIONS ************************/

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Area, arrow::DoubleBuilder, geo, 
    OGR_G_Area(geo));

UNARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Length, arrow::DoubleBuilder, geo, 
    OGR_G_Length(geo));

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Distance, arrow::DoubleBuilder, geo_1, geo_2,
    OGR_G_Distance(geo_1, geo_2));

 
/************************ SPATIAL RELATIONSHIP ************************/

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Equals, arrow::BooleanBuilder, geo_1, geo_2,
    OGR_G_Equals(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Touches, arrow::BooleanBuilder, geo_1, geo_2,
    OGR_G_Touches(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Overlaps, arrow::BooleanBuilder, geo_1, geo_2,
    OGR_G_Overlaps(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Crosses, arrow::BooleanBuilder, geo_1, geo_2,
    OGR_G_Crosses(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Contains, arrow::BooleanBuilder, geo_1, geo_2,
    OGR_G_Contains(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Intersects, arrow::BooleanBuilder, geo_1, geo_2,
    OGR_G_Intersects(geo_1, geo_2) != 0);

BINARY_WKT_FUNC_WITH_GDAL_IMPL_T1(
    ST_Within, arrow::BooleanBuilder, geo_1, geo_2,
    OGR_G_Within(geo_1, geo_2) != 0);


/*********************** AGGREGATE FUNCTIONS ***************************/

std::shared_ptr<arrow::Array>
ST_Union_Aggr(const std::shared_ptr<arrow::Array> &geometries){

    auto len = geometries->length();
    assert(len > 0);
    auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries);

    arrow::StringBuilder builder;
    OGRGeometry  *geo_result, *geo_var, *geo_tmp;
    char *wkt_result;

    CHECK_GDAL(OGRGeometryFactory::createFromWkt(
        wkt_geometries->GetString(0).c_str(), nullptr, &geo_result));

    for (int32_t i = 1; i < len; i++) {
        CHECK_GDAL(OGRGeometryFactory::createFromWkt(
            wkt_geometries->GetString(i).c_str(), nullptr, &geo_var));
        geo_tmp = geo_result;
        geo_result = geo_result->Union(geo_var);
        OGRGeometryFactory::destroyGeometry(geo_var);
        OGRGeometryFactory::destroyGeometry(geo_tmp);
    }

    CHECK_GDAL(OGR_G_ExportToWkt(geo_result, &wkt_result));
    OGRGeometryFactory::destroyGeometry((OGRGeometry*)geo_result);
    CHECK_ARROW(builder.Append(wkt_result));
    CPLFree(wkt_result);
    std::shared_ptr<arrow::Array> results;
    CHECK_ARROW(builder.Finish(&results));
    return results;
}



std::shared_ptr<arrow::Array>
ST_Envelope_Aggr(const std::shared_ptr<arrow::Array> &geometries){
    return ST_Envelope(ST_Union_Aggr(geometries));
}




} // gis
} // zilliz
