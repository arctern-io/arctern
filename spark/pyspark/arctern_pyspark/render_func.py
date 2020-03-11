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

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

import pyarrow as pa
import pandas as pd

def render_point_map(df, vega):
    schema = StructType([StructField('buffer', StringType(), True)])
    @pandas_udf(schema, PandasUDFType.MAP_ITER)
    def point_map_UDF(batch_iter, conf = vega):
        for pdf in batch_iter:
            pdf = pdf.drop_duplicates()
            arr_x = pa.array(pdf.x, type='uint32')
            arr_y = pa.array(pdf.y, type='uint32')
            from arctern_gis import point_map
            res = point_map(arr_x, arr_y, conf.encode('utf-8'))
            buffer = res.buffers()[1].hex()
            buf_df = pd.DataFrame([(buffer,)],["buffer"])
            yield buf_df

    hex_data = df.mapInPandas(point_map_UDF).collect()[0][0]
    return hex_data


def render_heat_map(df, vega):
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

    schema = StructType([StructField('buffer', StringType(), True)])
    @pandas_udf(schema, PandasUDFType.MAP_ITER)
    def heat_map_UDF(batch_iter, conf = vega):
        for pdf in batch_iter:
            arrs = pdf.groupby(['x','y'])['c'].agg(['sum']).reset_index()
            arrs.columns = ['x', 'y', 'c']
            arr_x = pa.array(arrs.x, type='uint32')
            arr_y = pa.array(arrs.y, type='uint32')
            arr_c = pa.array(arrs.c, type='uint32')
            from arctern_gis import heat_map
            res = heat_map(arr_x, arr_y, arr_c, conf.encode('utf-8'))
            buffer = res.buffers()[1].hex()
            buf_df = pd.DataFrame([(buffer,)],["buffer"])
            yield buf_df

    render_agg_df = df.mapInPandas(render_agg_UDF).coalesce(1)
    hex_data = render_agg_df.mapInPandas(heat_map_UDF).collect()[0][0]
    return hex_data


def render_choropleth_map(df, vega):
    agg_schema = StructType([StructField('wkt', StringType(), True),
                             StructField('c', IntegerType(), True)])
    @pandas_udf(agg_schema, PandasUDFType.MAP_ITER)
    def render_agg_UDF(batch_iter):
        for pdf in batch_iter:
            res = pdf.groupby(['wkt'])
            res = res['c'].agg(['sum']).reset_index()
            res.columns = ['wkt', 'c']
            yield res

    schema = StructType([StructField('buffer', StringType(), True)])
    @pandas_udf(schema, PandasUDFType.MAP_ITER)
    def choropleth_map_UDF(batch_iter, conf = vega):
        for pdf in batch_iter:
            arrs = pdf.groupby(['wkt'])['c'].agg(['sum']).reset_index()
            arrs.columns = ['wkt', 'c']
            arr_wkt = pa.array(arrs.wkt, type='string')
            arr_c = pa.array(arrs.c, type='uint32')
            from arctern_gis import choropleth_map
            res = choropleth_map(arr_wkt, arr_c, conf.encode('utf-8'))
            buffer = res.buffers()[1].hex()
            buf_df = pd.DataFrame([(buffer,)],["buffer"])
            yield buf_df

    render_agg_df = df.mapInPandas(render_agg_UDF).coalesce(1)
    hex_data = render_agg_df.mapInPandas(choropleth_map_UDF).collect()[0][0]
    return hex_data
