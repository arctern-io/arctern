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

# def _plot_spark_data_frame(ax, geoms):
#     import json
#     from pyspark.sql.functions import col
    
#     if len(geoms.columns) != 1:
#         raise RuntimeError(f"The input param 'geoms' should have only one column. geoms schema = {geoms.schema}")
#     head_row = geoms.head()
#     if head_row==None:
#         return ax
#     head_row = head_row[0]
#     if isinstance(head_row, str): # input is geojson
#         geoms = geoms.collect()
#     elif isinstance(head_row,bytearray): #input is wkb
#         from arctern_pyspark._wrapper_func import ST_AsGeoJSON
#         geoms = geoms.select(ST_AsGeoJSON(col(geoms.columns[0])))
#         geoms = geoms.collect()
#     else:
#         raise RuntimeError(f"unexpected input type, {type(head_row)}")
        
#     plot_collect = dict()
#     for geo in geoms:
#         geo_dict = json.loads(geo[0])
#         _flat_geoms(geo_dict,plot_collect)
    
#     _plot_collection(ax, plot_collect)


# def plot(ax, geoms):
#     import pyspark.sql.dataframe
#     if isinstance(geoms,pyspark.sql.dataframe.DataFrame):
#         _plot_spark_data_frame(ax, geoms)

def plot(ax, geoms):
    import pyspark.sql.dataframe
    import arctern
    if isinstance(geoms, pyspark.sql.dataframe.DataFrame):
        pandas_df = geoms.toPandas()
        arctern.plot(ax, pandas_df)