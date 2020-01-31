
__all__ = [
    "ST_Point",
    "point_map",
    "my_plot"
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


@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def my_plot(x, y):
    arr_x = pa.array(x, type='uint32')
    arr_y = pa.array(y, type='uint32')
    from zilliz_gis import point_map
    curve_z = point_map(arr_x, arr_y)
    curve_z_copy = curve_z
    curve_z = curve_z.buffers()[1].to_pybytes()
    return curve_z_copy.buffers()[1].hex()


