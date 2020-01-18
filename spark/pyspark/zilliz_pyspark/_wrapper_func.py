
__all__ = [
    "ST_Point",
    "point_map"
]

import pandas as pd
import pyarrow as pa
from pyspark.sql.functions import pandas_udf, PandasUDFType

def toArrow(parameter):
    return  pa.array(parameter)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Point(x, y):
    return 1
#    from zilliz_gis import make_point
   # x, y = toArrow(x), toArrow(y)
   # return pd.Series(make_point(x, y))


@pandas_udf("long", PandasUDFType.GROUPED_AGG)
def point_map(x, y):
    return 1
    #from zilliz_gis import point_map
   # x, y = toArrow(x), toArrow(y)
   # return pd.Series(point_map(x, y))

