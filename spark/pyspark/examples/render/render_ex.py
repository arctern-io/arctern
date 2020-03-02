# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pyspark.sql import SparkSession
from zilliz_pyspark import register_funcs
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from zilliz_gis.util.vega.heat_map.vega_heat_map import VegaHeatMap

import pyarrow as pa
import pandas as pd

def run_heat_map(spark):
    points_data = []
    for i in range(300):
        points_data.extend([(i, i, i)])
    df = spark.createDataFrame(data = points_data, schema = ["x", "y", "c"]).cache()
    vega_heat_map = VegaHeatMap(300, 200, 10.0)
    vega = vega_heat_map.build()

    schema = StructType([StructField('x', IntegerType(), True),
                        StructField('y', IntegerType(), True),
                        StructField('c', IntegerType(), True)])
    @pandas_udf(schema, PandasUDFType.MAP_ITER)
    def render_agg_UDF(batch_iter):
        for pdf in batch_iter:
            res = pdf.groupby(['x','y'])
            res = res['c'].agg(['sum']).reset_index()
            res.columns = ['x', 'y', 'c']
            yield res

    schema = StructType([StructField('buffer', StringType(), True)])
    @pandas_udf(schema, PandasUDFType.MAP_ITER)
    def heat_map_UDF(batch_iter, conf = vega):
        for pdf in batch_iter:
            arrs = pdf.groupby(['x','y'])['c'].agg(['sum']).reset_index()
            arrs.columns = ['x', 'y', 'c']
            arr_x = pa.array(arrs.x, type='uint32')
            arr_y = pa.array(arrs.y, type='uint32')
            arr_c = pa.array(arrs.c, type='uint32')
            from zilliz_gis import heat_map
            res = heat_map(arr_x, arr_y, arr_c, conf.encode('utf-8'))
            buffer = res.buffers()[1].hex()
            buf_df = pd.DataFrame([(buffer,)],["buffer"])
            yield buf_df

    render_agg_df = df.mapInPandas(render_agg_UDF).coalesce(1)
    hex_data = render_agg_df.mapInPandas(heat_map_UDF).collect()[0][0]

    import binascii
    binary_string = binascii.unhexlify(str(hex_data))
    with open('/tmp/hex_heat_map.png', 'wb') as png:
        png.write(binary_string)

if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python TestPointmap") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_heat_map(spark_session)

    spark_session.stop()

