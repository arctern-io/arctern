from arctern_spark import GeoSeries
import pytest
from databricks.koalas import Series
import numpy as np


s = GeoSeries(['Point (1 2)', None, np.nan])
print(s)
assert len(s) == 3
# assert s.hasnans
# assert s[1] is None
# assert s[2] is None

s1 = Series(['a', 'b', np.nan])
print(s1)
b = s1.hasnans

print(b)