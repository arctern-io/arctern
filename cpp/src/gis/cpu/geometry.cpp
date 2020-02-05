#include "geometry.h"

namespace zilliz {
namespace gis {


// std::shared_ptr<arrow::StringArray>
// ST_point(const arrow::DoubleArray &x,
//          const arrow::DoubleArray &y) {
// #ifdef GPU_IMPL
//   return detail::ST_point_gpu(x, y);
// #endif
// #ifdef CPU_SIMD_IMPL
//   return detail::ST_point_simd(x, y);
// #endif
// }

std::shared_ptr<arrow::Array>
ST_Point(const std::shared_ptr<arrow::Array> &point_x,
         const std::shared_ptr<arrow::Array> &point_y) {
    auto &point_arr_x = static_cast<const arrow::DoubleArray &>(*point_x);
    auto &point_arr_y = static_cast<const arrow::DoubleArray &>(*point_y);

    arrow::StringBuilder string_builder;
    std::shared_ptr<arrow::Array> array;
    OGRPoint point;

    for (int32_t i = 0; i < point_arr_x.length(); i++) {
        char *str=nullptr;
        point = OGRPoint(point_arr_x.Value(i), point_arr_y.Value(i));
        CHECK_GDAL(point.exportToWkt(&str));
        string_builder.Append(std::string(str));
        CPLFree(str);
    }

    string_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_Intersection(const std::shared_ptr<arrow::Array> &left_geometries,
                const std::shared_ptr<arrow::Array> &right_geometries) {
    auto &left_geometries_arr = static_cast<const arrow::StringArray &>(*left_geometries);
    auto &right_geometries_arr = static_cast<const arrow::StringArray &>(*right_geometries);

    arrow::StringBuilder string_builder;
    std::shared_ptr<arrow::Array> array;
    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < left_geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(left_geometries_arr.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(right_geometries_arr.GetString(i).c_str(), nullptr, &right_geo);
        auto inter_res = left_geo->Intersection(right_geo);
        char *str=nullptr;
        CHECK_GDAL(inter_res->exportToWkt(&str));
        string_builder.Append(std::string(str));
        CPLFree(str);
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);

    }

    string_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_IsValid(const std::shared_ptr<arrow::Array> &geometries) {
    auto &geometries_arr = static_cast<const arrow::StringArray &>(*geometries);

    arrow::BooleanBuilder bool_builder;
    std::shared_ptr<arrow::Array> array;

    OGRGeometry *geometry;
    for (int32_t i = 0; i < geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(geometries_arr.GetString(i).c_str(), nullptr, &geometry);
        geometry->IsValid() == 0 ? bool_builder.Append(false) : bool_builder.Append(true);
        OGRGeometryFactory::destroyGeometry(geometry);
    }

    bool_builder.Finish(&array);

    return array;
}


std::shared_ptr<arrow::Array>
ST_Equals(const std::shared_ptr<arrow::Array> &left_geometries,
          const std::shared_ptr<arrow::Array> &right_geometries) {
    auto &left_geometries_arr = static_cast<const arrow::StringArray &>(*left_geometries);
    auto &right_geometries_arr = static_cast<const arrow::StringArray &>(*right_geometries);

    arrow::BooleanBuilder bool_builder;
    std::shared_ptr<arrow::Array> array;
    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < left_geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(left_geometries_arr.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(right_geometries_arr.GetString(i).c_str(), nullptr, &right_geo);
        left_geo->Equals(right_geo) == 0 ? bool_builder.Append(false) : bool_builder.Append(true);
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);
    }

    bool_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_Touches(const std::shared_ptr<arrow::Array> &left_geometries,
           const std::shared_ptr<arrow::Array> &right_geometries) {
    auto &left_geometries_arr = static_cast<const arrow::StringArray &>(*left_geometries);
    auto &right_geometries_arr = static_cast<const arrow::StringArray &>(*right_geometries);

    arrow::BooleanBuilder bool_builder;
    std::shared_ptr<arrow::Array> array;
    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < left_geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(left_geometries_arr.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(right_geometries_arr.GetString(i).c_str(), nullptr, &right_geo);
        left_geo->Touches(right_geo) == 0 ? bool_builder.Append(false) : bool_builder.Append(true);
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);
    }

    bool_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_Overlaps(const std::shared_ptr<arrow::Array> &left_geometries,
            const std::shared_ptr<arrow::Array> &right_geometries) {
    auto &left_geometries_arr = static_cast<const arrow::StringArray &>(*left_geometries);
    auto &right_geometries_arr = static_cast<const arrow::StringArray &>(*right_geometries);

    arrow::BooleanBuilder bool_builder;
    std::shared_ptr<arrow::Array> array;
    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < left_geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(left_geometries_arr.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(right_geometries_arr.GetString(i).c_str(), nullptr, &right_geo);
        left_geo->Overlaps(right_geo) == 0 ? bool_builder.Append(false) : bool_builder.Append(true);
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);
    }

    bool_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_Crosses(const std::shared_ptr<arrow::Array> &left_geometries,
           const std::shared_ptr<arrow::Array> &right_geometries) {
    auto &left_geometries_arr = static_cast<const arrow::StringArray &>(*left_geometries);
    auto &right_geometries_arr = static_cast<const arrow::StringArray &>(*right_geometries);

    arrow::BooleanBuilder bool_builder;
    std::shared_ptr<arrow::Array> array;
    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < left_geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(left_geometries_arr.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(right_geometries_arr.GetString(i).c_str(), nullptr, &right_geo);
        left_geo->Crosses(right_geo) == 0 ? bool_builder.Append(false) : bool_builder.Append(true);
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);
    }

    bool_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_IsSimple(const std::shared_ptr<arrow::Array> &geometries) {
    auto &geometries_arr = static_cast<const arrow::StringArray &>(*geometries);

    arrow::BooleanBuilder bool_builder;
    std::shared_ptr<arrow::Array> array;

    OGRGeometry *geometry;

    for (int32_t i = 0; i < geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(geometries_arr.GetString(i).c_str(), nullptr, &geometry);
        geometry->IsSimple() == 0 ? bool_builder.Append(false) : bool_builder.Append(true);
        OGRGeometryFactory::destroyGeometry(geometry);
    }

    bool_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_PrecisionReduce(const std::shared_ptr<arrow::Array> &geometries, int32_t num_dot) {
    auto &geometries_arr = static_cast<const arrow::StringArray &>(*geometries);

    arrow::StringBuilder string_builder;
    std::shared_ptr<arrow::Array> array;

    // OGRGeometry *geometry;
    // OGRWktOptions options;
    // options.precision = num_dot;

    // for (int32_t i = 0; i < geometries_arr.length(); i++) {
    //     OGRGeometryFactory::createFromWkt(geometries_arr.GetString(i).c_str(), nullptr, &geometry);
    //     string_builder.Append(geometry->exportToWkt(options));
    // }
    for(int32_t i=0; i<geometries_arr.length(); ++i){
        string_builder.Append(geometries_arr.GetString(i));
    }

    string_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_GeometryType(const std::shared_ptr<arrow::Array> &geometries) {
    auto &geometries_arr = static_cast<const arrow::StringArray &>(*geometries);

    arrow::StringBuilder string_builder;
    std::shared_ptr<arrow::Array> array;

    OGRGeometry *geometry;
    for (int32_t i = 0; i < geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(geometries_arr.GetString(i).c_str(), nullptr, &geometry);
        string_builder.Append(geometry->getGeometryName());
        OGRGeometryFactory::destroyGeometry(geometry);
    }

    string_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_MakeValid(const std::shared_ptr<arrow::Array> &geometries) {
    auto &geometries_arr = static_cast<const arrow::StringArray &>(*geometries);
    std::cout << geometries_arr.GetString(0) << std::endl;
    arrow::StringBuilder string_builder;
    std::shared_ptr<arrow::Array> array;
    OGRGeometry *geometry;
    for (int32_t i = 0; i < geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(geometries_arr.GetString(i).c_str(), nullptr, &geometry);
        char *str=nullptr;
        CHECK_GDAL(geometry->MakeValid()->exportToWkt(&str))
        string_builder.Append(std::string(str));
        CPLFree(str);
        OGRGeometryFactory::destroyGeometry(geometry);
    }
    string_builder.Finish(&array);
    return array;
}


std::shared_ptr<arrow::Array>
ST_SimplifyPreserveTopology(const std::shared_ptr<arrow::Array> &geometries, double distanceTolerance) {
    auto &geometries_arr = static_cast<const arrow::StringArray &>(*geometries);

    arrow::StringBuilder string_builder;
    std::shared_ptr<arrow::Array> array;

    OGRGeometry *geometry;
    for (int32_t i = 0; i < geometries_arr.length(); i++) {
        OGRGeometryFactory::createFromWkt(geometries_arr.GetString(i).c_str(), nullptr, &geometry);
        char *str=nullptr;
        CHECK_GDAL(geometry->SimplifyPreserveTopology(distanceTolerance)->exportToWkt(&str))
        string_builder.Append(std::string(str));
        CPLFree(str);
        OGRGeometryFactory::destroyGeometry(geometry);
    }

    string_builder.Finish(&array);
    return array;
}


//ST_PolygonFromEnvelope
//std::shared_ptr<arrow::Array>
//ST_PolygonFromEnvelope(const std::shared_ptr<arrow::Array> min_x,
//                       const std::shared_ptr<arrow::Array> min_y,
//                       const std::shared_ptr<arrow::Array> max_x,
//                       const std::shared_ptr<arrow::Array> max_y) {
//  auto min_x_len = min_x->length();
//  auto min_y_len = min_y->length();
//  auto max_x_len = max_x->length();
//  auto max_y_len = max_y->length();
//  assert(min_x_len == min_y_len == max_x_len == max_y_len);
//
//  auto &min_x_arr = static_cast<const arrow::DoubleArray &>(*min_x);
//  auto &min_y_arr = static_cast<const arrow::DoubleArray &>(*min_y);
//  auto &max_x_arr = static_cast<const arrow::DoubleArray &>(*max_x);
//  auto &max_y_arr = static_cast<const arrow::DoubleArray &>(*max_y);
//
//  std::shared_ptr<arrow::Array> res_arr;
//  arrow::StringBuilder builder;
//
//  for (int32_t i = 0; i < min_x_len; i++) {
//    //TODO : contructs a polygon
//    //CHECK_ARROW(builder.Append(polygon.exportToWkt()));
//  }
//  CHECK_ARROW(builder.Finish(&res_arr));
//
//  return res_arr;
//}

std::shared_ptr<arrow::Array>
ST_Contains(const std::shared_ptr<arrow::Array> geo_arr1,
            const std::shared_ptr<arrow::Array> geo_arr2) {
    auto len1 = geo_arr1->length();
    auto len2 = geo_arr2->length();
    assert(len1 == len2);

    std::shared_ptr<arrow::Array> res_arr;
    arrow::BooleanBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr1);
    auto &geo_str_arr2 = static_cast<const arrow::StringArray &>(*geo_arr2);

    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < len1; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(geo_str_arr2.GetString(i).c_str(), nullptr, &right_geo);
        left_geo->Contains(right_geo) == 0 ? builder.Append(false) : builder.Append(true);
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_Intersects(const std::shared_ptr<arrow::Array> geo_arr1,
              const std::shared_ptr<arrow::Array> geo_arr2) {
    auto len1 = geo_arr1->length();
    auto len2 = geo_arr2->length();
    assert(len1 == len2);

    std::shared_ptr<arrow::Array> res_arr;
    arrow::BooleanBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr1);
    auto &geo_str_arr2 = static_cast<const arrow::StringArray &>(*geo_arr2);

    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < len1; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(geo_str_arr2.GetString(i).c_str(), nullptr, &right_geo);
        left_geo->Intersects(right_geo) == 0 ? builder.Append(false) : builder.Append(true);
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_Within(const std::shared_ptr<arrow::Array> geo_arr1,
          const std::shared_ptr<arrow::Array> geo_arr2) {
    auto len1 = geo_arr1->length();
    auto len2 = geo_arr2->length();
    assert(len1 == len2);

    std::shared_ptr<arrow::Array> res_arr;
    arrow::BooleanBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr1);
    auto &geo_str_arr2 = static_cast<const arrow::StringArray &>(*geo_arr2);

    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < len1; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(geo_str_arr2.GetString(i).c_str(), nullptr, &right_geo);
        left_geo->Within(right_geo) == 0 ? builder.Append(false) : builder.Append(true);
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_Distance(const std::shared_ptr<arrow::Array> geo_arr1,
            const std::shared_ptr<arrow::Array> geo_arr2) {
    auto len1 = geo_arr1->length();
    auto len2 = geo_arr2->length();
    assert(len1 == len2);

    std::shared_ptr<arrow::Array> res_arr;
    arrow::DoubleBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr1);
    auto &geo_str_arr2 = static_cast<const arrow::StringArray &>(*geo_arr2);

    OGRGeometry *left_geo;
    OGRGeometry *right_geo;

    for (int32_t i = 0; i < len1; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &left_geo);
        OGRGeometryFactory::createFromWkt(geo_str_arr2.GetString(i).c_str(), nullptr, &right_geo);
        CHECK_ARROW(builder.Append(left_geo->Distance(right_geo)));
        OGRGeometryFactory::destroyGeometry(left_geo);
        OGRGeometryFactory::destroyGeometry(right_geo);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_Area(const std::shared_ptr<arrow::Array> geo_arr) {
    auto len = geo_arr->length();

    std::shared_ptr<arrow::Array> res_arr;
    arrow::DoubleBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr);

    OGRGeometry *g;

    for (int32_t i = 0; i < len; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &g);
        //TODO, check g is class of OGRSurface, if not return 0
        CHECK_ARROW(builder.Append((reinterpret_cast<OGRSurface*>(g))->get_Area()));
        OGRGeometryFactory::destroyGeometry(g);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_Centroid(const std::shared_ptr<arrow::Array> geo_arr) {
    auto len = geo_arr->length();

    std::shared_ptr<arrow::Array> res_arr;
    arrow::StringBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr);

    OGRGeometry *geo;

    for (int32_t i = 0; i < len; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &geo);
        OGRPoint *poPoint = new OGRPoint();
        geo->Centroid(poPoint);
        char *str=nullptr;
        CHECK_GDAL(poPoint->exportToWkt(&str));
        builder.Append(std::string(str));
        CPLFree(str);
        OGRGeometryFactory::destroyGeometry(geo);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_Length(const std::shared_ptr<arrow::Array> geo_arr) {
    auto len = geo_arr->length();

    std::shared_ptr<arrow::Array> res_arr;
    arrow::DoubleBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr);

    OGRGeometry *g;

    for (int32_t i = 0; i < len; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &g);
        //TODO, check if g is class of OGRCurve, if not return 0
        CHECK_ARROW(builder.Append((reinterpret_cast<OGRCurve*>(g))->get_Length()));
        OGRGeometryFactory::destroyGeometry(g);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_ConvexHull(const std::shared_ptr<arrow::Array> geo_arr) {
    auto len = geo_arr->length();

    std::shared_ptr<arrow::Array> res_arr;
    arrow::StringBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr);

    OGRGeometry *geo;

    for (int32_t i = 0; i < len; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &geo);
        char *str=nullptr;
        CHECK_GDAL(geo->ConvexHull()->exportToWkt(&str));
        builder.Append(std::string(str));
        CPLFree(str);
        OGRGeometryFactory::destroyGeometry(geo);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_NPoints(const std::shared_ptr<arrow::Array> geo_arr) {
    auto len = geo_arr->length();

    std::shared_ptr<arrow::Array> res_arr;
    arrow::Int64Builder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr);

    OGRGeometry *g;

    for (int32_t i = 0; i < len; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &g);
        //check if g is type of OGRCurve, if not return 0
        CHECK_ARROW(builder.Append((reinterpret_cast<OGRCurve*>(g))->getNumPoints()));
        OGRGeometryFactory::destroyGeometry(g);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_Envelope(const std::shared_ptr<arrow::Array> geo_arr) {
    auto len = geo_arr->length();

    std::shared_ptr<arrow::Array> res_arr;
    arrow::StringBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr);

    OGRGeometry *geo;

    for (int32_t i = 0; i < len; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &geo);
        char *str = nullptr;
        CHECK_GDAL(geo->Boundary()->exportToWkt(&str));
        builder.Append(std::string(str));
        CPLFree(str);
        OGRGeometryFactory::destroyGeometry(geo);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}

std::shared_ptr<arrow::Array>
ST_Buffer(const std::shared_ptr<arrow::Array> geo_arr, double dfDist) {
    auto len = geo_arr->length();

    std::shared_ptr<arrow::Array> res_arr;
    arrow::StringBuilder builder;

    auto &geo_str_arr1 = static_cast<const arrow::StringArray &>(*geo_arr);

    OGRGeometry *geo;

    for (int32_t i = 0; i < len; i++) {
        OGRGeometryFactory::createFromWkt(geo_str_arr1.GetString(i).c_str(), nullptr, &geo);
        char *str = nullptr;
        CHECK_GDAL(geo->Buffer(dfDist)->exportToWkt(&str));
        builder.Append(std::string(str));
        CPLFree(str);
        OGRGeometryFactory::destroyGeometry(geo);
    }
    CHECK_ARROW(builder.Finish(&res_arr));
    return res_arr;
}
} // gis
} // zilliz