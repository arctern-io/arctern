from osgeo import ogr
from arctern_spark.geoseries import GeoSeries


# data = GeoSeries(["POINT (1.333 2.666)", "POINT (2.655 4.447)"])
# rst = data.precision_reduce(3).to_wkt()
# assert rst[0] == "POINT (1.33 2.67)"
# assert rst[1] == "POINT (2.66 4.45)"

from pandas import Series
import pandas as pd
pd.set_option("max_colwidth", 100000)
data = ["POLYGON((0 0,1 0,1 1,0 0))"]
data = GeoSeries(data)
rst = data.buffer(1.2).to_wkt()

print(rst)