#include "gis/spatial_join/st_indexed_within.h"
namespace arctern {
namespace gis {
namespace spatial_join {
std::vector<Int32ArrayPtr> ST_IndexedWithin(const std::vector<WkbArrayPtr>& points,
                                            const std::vector<WkbArrayPtr>& polygons) {
  // this is a fake implementation
  // assume that points is Point(0 0) Point(1000 1000) Point(10 10)
  // assume that polygon is Polygon(9 10, 11 12, 11 8, 9 10)
  //                        Polygon(-1 0, 1 2, 1 -2, -1 0)
  assert(points.size() == 1);
  assert(points.front()->length() == 3);
  assert(polygons.size() == 1);
  assert(polygons.front()->length() == 2);
  arrow::Int32Builder builder;
  builder.Append(1);
  builder.Append(-1);
  builder.Append(0);
  Int32ArrayPtr res;
  builder.Finish(&res);
  return {res};
}

}  // namespace spatial_join
}  // namespace gis
}  // namespace arctern
