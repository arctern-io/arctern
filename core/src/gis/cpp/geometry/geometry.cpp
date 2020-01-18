#include "gis/cpp/geometry/utils.h"
#include "gis/cpp/geometry/geometry.h"

namespace zilliz {
namespace gis {
namespace cpp {
namespace gemetry {

typedef boost::geometry::model::d2::point_xy<double> point_2d;

std::shared_ptr<arrow::Array>
ST_point(const double *const ptr_x,
         const double *const ptr_y,
         const int64_t len) {
  std::shared_ptr<arrow::Array> point_arr;
  arrow::StringBuilder builder;

//  for (int64_t i = 0; i < len; i++) {
//    std::string point_wkt_str = boost::lexical_cast<std::string>(
//        "POINT(" + std::to_string(*((double *) (ptr_x + i))) + " " + std::to_string(*((double *) (ptr_y + i))) + ")");
//    CHECK_STATUS(builder.Append(point_wkt_str));
//  }
  for (int64_t i = 0; i < len; i++) {
    auto point2d = point_2d(*((double *) (ptr_x + i)), *((double *) (ptr_y + i)));
    auto point_wkt = boost::geometry::wkt(point2d);
    auto point_wkt_str = boost::lexical_cast<std::string>(point_wkt);
    CHECK_STATUS(builder.Append(point_wkt_str));
  }
  CHECK_STATUS(builder.Finish(&point_arr));

  return point_arr;
}

std::shared_ptr<arrow::Array>
ST_point(std::shared_ptr<arrow::Array> ptr_x, std::shared_ptr<arrow::Array> ptr_y){
  auto double_ptr_x = reinterpret_cast<const double *>(ptr_x->data()->buffers[1].get()->data());
  auto double_ptr_y = reinterpret_cast<const double *>(ptr_y->data()->buffers[1].get()->data());
  auto length = ptr_x->length();
  return ST_point( double_ptr_x,double_ptr_y,length);
}

} // geometry
} // cpp
} // gis
} // zilliz