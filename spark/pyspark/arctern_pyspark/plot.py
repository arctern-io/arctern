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

def plot(ax, geoms, **style_kwds):
    """
     Plots a collection of geometries to `ax`

     :type ax: matplotlib.axes.Axes
     :parms ax: The axes where geometries will be plotted

     :type geoms: pyspark.sql.dataframe.DataFrame
     :parms geoms: sequence of geometries

     :type **style_kwds: dict
     :parms **style_kwds: optional, collection of plot style
         `Polygon` and `MultiPolygon`:
             linewidth
             linestyle
             edgecolor
             facecolor
         `LineString` and `MultiLineString`:
             color
             linewidth
             linestyle
         `Point` and `MultiPoint`:
             color
             marker
             markersize
     :example:
         from arctern_pyspark import register_funcs
         from pyspark.sql.types import *
         import matplotlib.pyplot as plt
         from arctern_pyspark import plot

         spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
         raw_data = []
         raw_data.append(('point(0 0)',))
         raw_data.append(('linestring(0 10, 5 5, 10 0)',))
         raw_data.append(('polygon((2 2,2 3,3 3,3 2,2 2))',))
         raw_data.append(("GEOMETRYCOLLECTION(" \
                         "polygon((1 1,1 2,2 2,2 1,1 1))," \
                         "linestring(0 1, 5 6, 10 11)," \
                         "POINT(4 7))",))
         raw_schema = StructType([StructField('geo', StringType(),False)])

         df = spark.createDataFrame(data=raw_data, schema=raw_schema)
         df.createOrReplaceTempView("geoms")

         df2=spark.sql("select st_geomfromtext(geo) from geoms")

         fig, ax = plt.subplots()
         plot(ax, df2,
              color=['orange', 'green'],
              marker='^',
              markersize=100,
              linewidth=[None, 7, 8],
              linestyle=[None, 'dashed', 'dashdot'],
              edgecolor=[None, None, 'red'],
              facecolor=[None, None, 'black'])
         ax.grid()
         fig.savefig('/tmp/plot_test.png')
    """
    import pyspark.sql.dataframe
    import arctern
    if isinstance(geoms, pyspark.sql.dataframe.DataFrame):
        pandas_df = geoms.toPandas()
        arctern.plot(ax, pandas_df, **style_kwds)
