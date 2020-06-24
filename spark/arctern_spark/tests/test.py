from osgeo import ogr
from arctern_spark.geoseries import GeoSeries


data = GeoSeries(["POINT (1.333 2.666)", "POINT (2.655 4.447)"])
# rst = data.precision_reduce(3).to_wkt()
# assert rst[0] == "POINT (1.33 2.67)"
# assert rst[1] == "POINT (2.66 4.45)"

print(data[0])

data1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
data2 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
rst = data1.geom_equals(data2)
assert len(rst) == 2
assert rst[0] == 1
assert rst[1] == 0

data1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
data2 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
rst = data1.geom_equals(data2)
assert len(rst) == 2
assert rst[0] == 1
assert rst[1] == 0

s0 = GeoSeries("POLYGON ((1 1,1 2,2 2,2 1,1 1))")[0]
rst = data2.geom_equals(s0)
print(rst)
assert len(rst) == 2
assert rst[0] == 1
assert rst[1] == 0