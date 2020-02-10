
__all__ = [
    "ST_Point_UDF",
    "my_plot"
]

import pandas as pd
import pyarrow as pa
from pyspark.sql.functions import pandas_udf, PandasUDFType

def toArrow(parameter):
    return  pa.array(parameter)

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def my_plot(x, y):
    arr_x = pa.array(x, type='uint32')
    arr_y = pa.array(y, type='uint32')
    from zilliz_gis import point_map
    curve_z = point_map(arr_x, arr_y)
    curve_z_copy = curve_z
    curve_z = curve_z.buffers()[1].to_pybytes()
    return curve_z_copy.buffers()[1].hex()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Point_UDF(x, y):
    arr_x = pa.array(x, type='double')
    arr_y = pa.array(y, type='double')
    from zilliz_gis import ST_Point
    point_arr = ST_Point(arr_x, arr_y)
    return point_arr.to_pandas()

