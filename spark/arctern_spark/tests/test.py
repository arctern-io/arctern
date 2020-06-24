from osgeo import ogr
from arctern_spark.geoseries import GeoSeries


# data = GeoSeries(["POINT (1.333 2.666)", "POINT (2.655 4.447)"])
# rst = data.precision_reduce(3).to_wkt()
# assert rst[0] == "POINT (1.33 2.67)"
# assert rst[1] == "POINT (2.66 4.45)"

from pandas import Series

data1 = [1.3, 2.5]
data2 = [3.8, 4.9]
string_ptr = GeoSeries.point(data1, data2).to_wkt()
assert len(string_ptr) == 2
assert string_ptr[0] == "POINT (1.3 3.8)"
assert string_ptr[1] == "POINT (2.5 4.9)"

string_ptr = GeoSeries.point(Series([1, 2], dtype='double'), 5).to_wkt()
assert len(string_ptr) == 2
assert string_ptr[0] == "POINT (1 5)"
assert string_ptr[1] == "POINT (2 5)"

string_ptr = GeoSeries.point(5, Series([1, 2], dtype='double')).to_wkt()
assert len(string_ptr) == 2
assert string_ptr[0] == "POINT (5 1)"
assert string_ptr[1] == "POINT (5 2)"

string_ptr = GeoSeries.point(5.0, 1.0).to_wkt()
assert len(string_ptr) == 1
assert string_ptr[0] == "POINT (5 1)"