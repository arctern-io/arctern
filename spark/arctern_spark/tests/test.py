from osgeo import ogr
from arctern_spark.geoseries import GeoSeries


data = GeoSeries(["POINT (1.333 2.666)", "POINT (2.655 4.447)"])
rst = data.precision_reduce(3).to_wkt()
assert rst[0] == "POINT (1.33 2.67)"
assert rst[1] == "POINT (2.66 4.45)"