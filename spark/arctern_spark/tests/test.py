from arctern_spark import GeoSeries as spark_series
import pytest
from databricks.koalas import Series
import numpy as np

import numpy as np
from arctern.geoseries import GeoSeries as python_series


# s1 = spark_series(['Point (1 2)', 'Point (2 2)', 'Point (3 2)', 'Point (4 2)', 'Point (5 2)', None, np.nan])
# s2 = python_series(['Point (1 2)', None, np.nan])
# s3 = Series(['1', '2', '3', '4', '5', '6', '7'])
# r1 = s1.hasnans
# r2 = s2.hasnans
# r3 = s3.hasnans
# print(r1)
# print(r2)
# print(r3)
import base64
index = ['a', 'b', 'c', 'd']

# property
data = spark_series(["LINESTRING (0 0, 1 0, 1 1, 0 1, 0 0)", "POLYGON ((0 0,1 0,1 1,0 1,0 0))"])
# rst = data.exterior
print(data.to_wkt())
# print(type(rst))
# rst2 = rst.to_wkt()
# print(rst2)
# r = s.fillna(a)
# print(r)
# print(r)
# print(s)
# print(s['a'])
# print(a)

# print(type(r2))
# print(type(r3))
# r1 = s1.loc[1:]
# print(r1)


# r = s.loc[1:]
# print(r)
# s2 = GeoSeries(r)
# print(type(r))
# print(s2[1:])