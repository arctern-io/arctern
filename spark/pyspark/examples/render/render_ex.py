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

from arctern_gis.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d
from arctern_gis.util.vega.heat_map.vega_heat_map import VegaHeatMap
from arctern_gis.util.vega.choropleth_map.choropleth_map import VegaChoroplethMap

from arctern_pyspark import heatmap
from arctern_pyspark import pointmap
from arctern_pyspark import choroplethmap

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
    hex_data = pointmap(df, vega) 
    save_png(hex_data, '/tmp/hex_point_map.png')

def run_heat_map(spark):
    points_data = []
    for i in range(300):
        points_data.extend([(i, i, i)])
    df = spark.createDataFrame(data = points_data, schema = ["x", "y", "c"]).cache()
    vega_heat_map = VegaHeatMap(300, 200, 10.0)
    vega = vega_heat_map.build()
    hex_data = heatmap(df, vega) 
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
    res = choroplethmap(df, vega)
    save_png(res, '/tmp/hex_choropleth_map.png')

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
    