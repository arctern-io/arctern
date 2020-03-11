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
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from zilliz_gis.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d
from zilliz_gis.util.vega.heat_map.vega_heat_map import VegaHeatMap
from zilliz_gis.util.vega.choropleth_map.choropleth_map import VegaChoroplethMap
from zilliz_pyspark import register_funcs
from zilliz_pyspark import point_map_UDF
import pyarrow as pa
import pandas as pd

def save_png(hex_data, file_name):
    import binascii
    binary_string = binascii.unhexlify(str(hex_data))
    with open(file_name, 'wb') as png:
        png.write(binary_string)


def run_point_map(spark):
    points_data = []
    for i in range(300):
        points_data.extend([(i, i)])
    df = spark.createDataFrame(data = points_data, schema = ["x", "y"]).cache().coalesce(1)
    vega_circle2d = VegaCircle2d(300, 200, 3, "#2DEF4A", 0.5)
    vega = vega_circle2d.build()
    register_funcs(spark)

#    schema = StructType([StructField('buffer', StringType(), True)])
#    @pandas_udf(schema, PandasUDFType.MAP_ITER)
#    def point_map_UDF(batch_iter, conf = vega):
#        for pdf in batch_iter:
#            pdf = pdf.drop_duplicates()
#            arr_x = pa.array(pdf.x, type='uint32')
#            arr_y = pa.array(pdf.y, type='uint32')
#            from zilliz_gis import point_map
#            res = point_map(arr_x, arr_y, conf.encode('utf-8'))
#            buffer = res.buffers()[1].hex()
#            buf_df = pd.DataFrame([(buffer,)],["buffer"])
#            yield buf_df

    hex_data = df.mapInPandas(point_map_UDF("vega")).collect()[0][0]
    save_png(hex_data, '/tmp/hex_point_map.png')


def run_heat_map(spark):
    points_data = []
    for i in range(300):
        points_data.extend([(i, i, i)])
    df = spark.createDataFrame(data = points_data, schema = ["x", "y", "c"]).cache()
    vega_heat_map = VegaHeatMap(300, 200, 10.0)
    vega = vega_heat_map.build()

    agg_schema = StructType([StructField('x', IntegerType(), True),
                             StructField('y', IntegerType(), True),
                             StructField('c', IntegerType(), True)])
    @pandas_udf(agg_schema, PandasUDFType.MAP_ITER)
    def render_agg_UDF(batch_iter):
        for pdf in batch_iter:
            res = pdf.groupby(['x','y'])
            res = res['c'].agg(['sum']).reset_index()
            res.columns = ['x', 'y', 'c']
            yield res

#    schema = StructType([StructField('buffer', StringType(), True)])
#    @pandas_udf(schema, PandasUDFType.MAP_ITER)
#    def heat_map_UDF(batch_iter, conf = vega):
#        for pdf in batch_iter:
#            arrs = pdf.groupby(['x','y'])['c'].agg(['sum']).reset_index()
#            arrs.columns = ['x', 'y', 'c']
#            arr_x = pa.array(arrs.x, type='uint32')
#            arr_y = pa.array(arrs.y, type='uint32')
#            arr_c = pa.array(arrs.c, type='uint32')
#            from zilliz_gis import heat_map
#            res = heat_map(arr_x, arr_y, arr_c, conf.encode('utf-8'))
#            buffer = res.buffers()[1].hex()
#            buf_df = pd.DataFrame([(buffer,)],["buffer"])
#            yield buf_df

    render_agg_df = df.mapInPandas(render_agg_UDF).coalesce(1)
    hex_data = render_agg_df.mapInPandas(heat_map_UDF).collect()[0][0]
    save_png(hex_data, '/tmp/hex_heat_map.png')


def run_choropleth_map(spark):
    points_data = []
    points_data.extend([("POLYGON (("
                         "-73.98128 40.754771, "
                         "-73.980185 40.754771, "
                         "-73.980185 40.755587, "
                         "-73.98128 40.755587, "
                         "-73.98128 40.754771))", 5.0)])
    df = spark.createDataFrame(data = points_data, schema = ["wkt", "c"]).cache()
    vega_choropleth_map = VegaChoroplethMap(1900, 1410,
                                            [-73.984092, 40.753893, -73.977588, 40.756342],
                                            "blue_to_red", [2.5, 5], 1.0)
    vega = vega_choropleth_map.build()

    agg_schema = StructType([StructField('wkt', StringType(), True),
                             StructField('c', IntegerType(), True)])
    @pandas_udf(agg_schema, PandasUDFType.MAP_ITER)
    def render_agg_UDF(batch_iter):
        for pdf in batch_iter:
            res = pdf.groupby(['wkt'])
            res = res['c'].agg(['sum']).reset_index()
            res.columns = ['wkt', 'c']
            yield res

#    schema = StructType([StructField('buffer', StringType(), True)])
#    @pandas_udf(schema, PandasUDFType.MAP_ITER)
#    def choropleth_map_UDF(batch_iter, conf = vega):
#        for pdf in batch_iter:
#            arrs = pdf.groupby(['wkt'])['c'].agg(['sum']).reset_index()
#            arrs.columns = ['wkt', 'c']
#            arr_wkt = pa.array(arrs.wkt, type='string')
#            arr_c = pa.array(arrs.c, type='uint32')
#            from zilliz_gis import choropleth_map
#            res = choropleth_map(arr_wkt, arr_c, conf.encode('utf-8'))
#            buffer = res.buffers()[1].hex()
#            buf_df = pd.DataFrame([(buffer,)],["buffer"])
#            yield buf_df

    render_agg_df = df.mapInPandas(render_agg_UDF).coalesce(1)
    hex_data = render_agg_df.mapInPandas(choropleth_map_UDF).collect()[0][0]
    save_png(hex_data, '/tmp/hex_choropleth_map.png')

if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python TestPointmap") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_point_map(spark_session)
    run_heat_map(spark_session)
    run_choropleth_map(spark_session)

    spark_session.stop()
