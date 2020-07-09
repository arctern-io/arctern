import pandas as pd
from arctern_spark import GeoSeries

import math
p1 = "POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0))"
p2 = "POINT (2 3)"
p3 = "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))"

geo_s = GeoSeries([p1, p2, None, p3])
pd_s = pd.Series([p1, p2, None, p3])

geo_res = geo_s.keys()
pd_res = pd_s.keys()

print(geo_res)
print(pd_res)
print(geo_res == pd_res)
# assert s1.geom_equals(s3).all()
# assert rst[0] == p1