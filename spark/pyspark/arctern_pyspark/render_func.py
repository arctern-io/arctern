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

import pyarrow as pa
from pyspark.sql.functions import pandas_udf, PandasUDFType

from pyspark.sql.types import *

def save_png_2D(hex_data, file_name):
    import binascii
    binary_string = binascii.unhexlify(str(hex_data))
    with open(file_name, 'wb') as png:
        png.write(binary_string)

def print_partitions(df):
    numPartitions = df.rdd.getNumPartitions()
    print("Total partitions: {}".format(numPartitions))
    print("Partitioner: {}".format(df.rdd.partitioner))
    df.explain()
    parts = df.rdd.glom().collect()
    i = 0
    j = 0
    for p in parts:
        print("Partition {}:".format(i))
        for r in p:
            print("Row {}:{}".format(j, r))
            j = j + 1
        i = i + 1

def pointmap_2D(df, vega):
    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def pointmap_wkt(point, conf=vega):
        arr_point = pa.array(point, type='string')
        from arctern_gis import point_map_wkt
        png = point_map_wkt(arr_point, conf.encode('utf-8'))
        buffer = png.buffers()[1].hex()
        return buffer

    df = df.coalesce(1)
    hex_data = df.agg(pointmap_wkt(df['point'])).collect()[0][0]
    return hex_data

def heatmap_2D(df, vega):
    agg_schema = StructType([StructField('point', StringType(), True),
                             StructField('w', IntegerType(), True)])

    @pandas_udf(agg_schema, PandasUDFType.MAP_ITER)
    def render_agg_UDF(batch_iter):
        for pdf in batch_iter:
            dd = pdf.groupby(['point'])
            dd = dd['w'].agg(['sum']).reset_index()
            dd.columns = ['point', 'w']
            yield dd

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def heatmap_wkt(point, w, conf=vega):
        arr_point = pa.array(point, type='string')
        arr_c = pa.array(w, type='int64')
        from arctern_gis import heat_map_wkt
        png = heat_map_wkt(arr_point, arr_c, conf.encode('utf-8'))
        buffer = png.buffers()[1].hex()
        return buffer

    first_agg_df = df.mapInPandas(render_agg_UDF).coalesce(1)
    final_agg_df = first_agg_df.mapInPandas(render_agg_UDF).coalesce(1)
    hex_data = final_agg_df.agg(heatmap_wkt(final_agg_df['point'], final_agg_df['w'])).collect()[0][0]
    return hex_data

def choroplethmap_2D(df, vega):
    agg_schema = StructType([StructField('wkt', StringType(), True),
                             StructField('w', IntegerType(), True)])

    @pandas_udf(agg_schema, PandasUDFType.MAP_ITER)
    def render_agg_UDF(batch_iter):
        for pdf in batch_iter:
            dd = pdf.groupby(['wkt'])
            dd = dd['w'].agg(['sum']).reset_index()
            dd.columns = ['wkt', 'w']
            yield dd

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def choroplethmap_wkt(wkt, w, conf=vega):
        arr_wkt = pa.array(wkt, type='string')
        arr_c = pa.array(w, type='int64')
        from arctern_gis import choropleth_map
        png = choropleth_map(arr_wkt, arr_c, conf.encode('utf-8'))
        buffer = png.buffers()[1].hex()
        return buffer

    first_agg_df = df.mapInPandas(render_agg_UDF).coalesce(1)
    final_agg_df = first_agg_df.mapInPandas(render_agg_UDF).coalesce(1)
    hex_data = final_agg_df.agg(choroplethmap_wkt(final_agg_df['wkt'], final_agg_df['w'])).collect()[0][0]
    return hex_data
