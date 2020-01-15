#include "gis/cpp/geometry/utils.h"
#include "gis/cpp/geometry/points.h"

namespace zilliz {
namespace gis {
namespace cpp {
namespace gemetry {

typedef boost::geometry::model::d2::point_xy<double> point_2d;

std::shared_ptr<arrow::Array>
ST_make_point(const double *const ptr_x,
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
ST_make_point(std::shared_ptr<arrow::Array> arr_x,
              std::shared_ptr<arrow::Array> arr_y) {

  auto d_arr_x = std::static_pointer_cast<arrow::DoubleArray>(arr_x);
  auto d_arr_y = std::static_pointer_cast<arrow::DoubleArray>(arr_y);

#ifdef demoDebug
  std::cout << "arr_x array :" << std::endl;
  auto arr_view = arr_x->ToString();
  std::cout << arr_view << std::endl;
#endif

  std::shared_ptr<arrow::Array> res;

  int64_t length = arr_x->length();
  assert(length == arr_y->length());

  arrow::StringBuilder builder;

  for (int64_t i = 0; i < length; ++i) {
    auto x = d_arr_x->Value(i);
    auto y = d_arr_y->Value(i);

    auto point2d = point_2d(x, y);
    auto point_wkt = boost::geometry::wkt(point2d);
    auto point_wkt_str = boost::lexical_cast<std::string>(point_wkt);
    CHECK_STATUS(builder.Append(point_wkt_str));
  }
  CHECK_STATUS(builder.Finish(&res));

  return res;
}

} // geometry
} // cpp
} // gis
} // zilliz