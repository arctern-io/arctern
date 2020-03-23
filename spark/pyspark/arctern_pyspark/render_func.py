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

__all__ = [
    "pointmap",
    "heatmap",
    "choroplethmap",
]

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

def pointmap(df, vega):
    from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit
    from ._wrapper_func import TransformAndProjection
    coor = vega.coor()
    bounding_box = vega.bounding_box()
    height = vega.height()
    width = vega.width()
    top_left = 'POINT (' + str(bounding_box[0]) +' '+ str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) +' '+ str(bounding_box[1]) + ')'
    if coor != 'EPSG:3857':
        df = df.select(TransformAndProjection(col('point'), lit(str(coor)), lit('EPSG:3857'), lit(top_left), lit(bottom_right), lit(int(height)), lit(int(width))).alias("point"))
    
    vega = vega.build()
    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def pointmap_wkb(point, conf=vega):
        from arctern import point_map_wkb
        return point_map_wkb(point, conf.encode('utf-8'))

    df = df.coalesce(1)
    hex_data = df.agg(pointmap_wkb(df['point'])).collect()[0][0]
    return hex_data

def heatmap(df, vega):
    from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col
    from pyspark.sql.types import (StructType, StructField, BinaryType, StringType, IntegerType)
    from ._wrapper_func import TransformAndProjection
    coor = vega.coor()
    bounding_box = vega.bounding_box()
    height = vega.height()
    width = vega.width()
    top_left = 'POINT (' + str(bounding_box[0]) +' '+ str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) +' '+ str(bounding_box[1]) + ')'
    if coor != 'EPSG:3857':
        df = df.select(TransformAndProjection(col('point'), lit(str(coor)), lit('EPSG:3857'), lit(top_left), lit(bottom_right), lit(int(height)), lit(int(width))).alias("point"), col('w'))
    
    vega = vega.build()
    agg_schema = StructType([StructField('point', BinaryType(), True),
                             StructField('w', IntegerType(), True)])

    @pandas_udf(agg_schema, PandasUDFType.MAP_ITER)
    def render_agg_UDF(batch_iter):
        for pdf in batch_iter:
            dd = pdf.groupby(['point'])
            dd = dd['w'].agg(['sum']).reset_index()
            dd.columns = ['point', 'w']
            yield dd

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def heatmap_wkb(point, w, conf=vega):
        from arctern import heat_map_wkb
        return heat_map_wkb(point, w, conf.encode('utf-8'))

    agg_df = df.mapInPandas(render_agg_UDF)
    agg_df = agg_df.coalesce(1)
    hex_data = agg_df.agg(heatmap_wkb(agg_df['point'], agg_df['w'])).collect()[0][0]
    return hex_data

def choroplethmap(df, vega):
    from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit
    from pyspark.sql.types import (StructType, StructField, BinaryType, StringType, IntegerType)
    from ._wrapper_func import TransformAndProjection
    coor = vega.coor()
    bounding_box = vega.bounding_box()
    height = vega.height()
    width = vega.width()
    top_left = 'POINT (' + str(bounding_box[0]) +' '+ str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) +' '+ str(bounding_box[1]) + ')'
    if (coor != 'EPSG:3857'):
        df = df.select(TransformAndProjection(col('wkt'), lit(str(coor)), lit('EPSG:3857'), lit(top_left), lit(bottom_right), lit(int(height)), lit(int(width))).alias("wkb"), col('w'))
    
    vega = vega.build()
    agg_schema = StructType([StructField('wkb', BinaryType(), True),
                             StructField('w', IntegerType(), True)])

    @pandas_udf(agg_schema, PandasUDFType.MAP_ITER)
    def render_agg_UDF(batch_iter):
        for pdf in batch_iter:
            dd = pdf.groupby(['wkb'])
            dd = dd['w'].agg(['sum']).reset_index()
            dd.columns = ['wkb', 'w']
            yield dd

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def choroplethmap_wkb(wkb, w, conf=vega):
        from arctern import choropleth_map
        return choropleth_map(wkb, w, conf.encode('utf-8'))

    @pandas_udf("double", PandasUDFType.GROUPED_AGG)
    def sum_udf(v):
        return v.sum()

    agg_df = df.where("wkb != ''")
    agg_df = agg_df.mapInPandas(render_agg_UDF)
    agg_df = agg_df.coalesce(1)
    hex_data = agg_df.agg(choroplethmap_wkb(agg_df['wkb'], agg_df['w'])).collect()[0][0]
    return hex_data
