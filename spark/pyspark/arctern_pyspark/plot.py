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
    Plot a collection of geometries to `ax`. Parameters 'linewidth', 'linestyle', 'edgecolor',
    'facecolor', 'color', 'marker', 'markersize' are used to describe the style of plotted figure.

    For geometry types `Polygon` and `MultiPolygon`, only 'linewidth', 'linestyle', 'edgecolor',
    'facecolor' are effective.

    For geometry types `Linestring` and `MultiLinestring`, only 'color', 'linewidth', 'linestyle' are effective.

    For geometry types `Point` and `MultiPoint`, only 'color', 'marker', 'markersize' are effective.

    :type ax: matplotlib.axes.Axes
    :param ax: The axes where geometries will be plotted.

    :type geoms: Series or DataFrame
    :param geoms: sequence of geometries.

    :type linewidth: list(float)
    :param linewidth: The width of line, the default value is 1.0.

    :type linestyle: list(string)
    :param linestyle: The style of the line， the default value is '-'.

    :type edgecolor: list(string)
    :param edgecolor: The edge color of the geometry, the default value is 'black'.

    :type facecolor: list(string)
    :param facecolor: The color of the face of the geometry, the default value is 'C0'.

    :type color: list(string)
    :param color: The color of the geometry, the default value is 'C0'.

    :type marker: string
    :param marker: The shape of point, the default value is 'o'.

    :type markersize: double
    :param markersize: The size of points, the default value is 6.0.

    :example:
       >>> from arctern_pyspark import register_funcs
       >>> from pyspark.sql.types import *
       >>> import matplotlib.pyplot as plt
       >>> from arctern_pyspark import plot
       >>> spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
       >>> raw_data = []
       >>> raw_data.append(('point(0 0)',))
       >>> raw_data.append(('linestring(0 10, 5 5, 10 0)',))
       >>> raw_data.append(('polygon((2 2,2 3,3 3,3 2,2 2))',))
       >>> raw_data.append(("GEOMETRYCOLLECTION("
                           "polygon((1 1,1 2,2 2,2 1,1 1)),"
                           "linestring(0 1, 5 6, 10 11),"
                           "POINT(4 7))",))
       >>> raw_schema = StructType([StructField('geo', StringType(),False)])
       >>> df = spark.createDataFrame(data=raw_data, schema=raw_schema)
       >>> df.createOrReplaceTempView("geoms")
       >>> df2=spark.sql("select st_geomfromtext(geo) from geoms")
       >>> fig, ax = plt.subplots()
       >>> plot(ax, df2,
                 color=['orange', 'green'],
                 marker='^',
                 markersize=100,
                 linewidth=[None, 7, 8],
                 linestyle=[None, 'dashed', 'dashdot'],
                 edgecolor=[None, None, 'red'],
                 facecolor=[None, None, 'black'])
       >>> ax.grid()
       >>> fig.savefig('/tmp/plot_test.png')
    """
    import pyspark.sql.dataframe
    import arctern
    if isinstance(geoms, pyspark.sql.dataframe.DataFrame):
        pandas_df = geoms.toPandas()
        arctern.plot.plot_geometry(ax, pandas_df, **style_kwds)
