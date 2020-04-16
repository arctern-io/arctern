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
    "ST_Point",
    "ST_Intersection",
    "ST_IsValid",
    "ST_PrecisionReduce",
    "ST_Equals",
    "ST_Touches",
    "ST_Overlaps",
    "ST_Crosses",
    "ST_IsSimple",
    "ST_GeometryType",
    "ST_MakeValid",
    "ST_SimplifyPreserveTopology",
    "ST_PolygonFromEnvelope",
    "ST_Contains",
    "ST_Intersects",
    "ST_Within",
    "ST_Distance",
    "ST_DistanceSphere",
    "ST_Area",
    "ST_Centroid",
    "ST_Length",
    "ST_HausdorffDistance",
    "ST_ConvexHull",
    "ST_NPoints",
    "ST_Envelope",
    "ST_Buffer",
    "ST_Union_Aggr",
    "ST_Envelope_Aggr",
    "ST_Transform",
    "ST_CurveToLine",
    "ST_GeomFromGeoJSON",
    "ST_PointFromText",
    "ST_PolygonFromText",
    "ST_LineStringFromText",
    "ST_GeomFromText",
    "ST_GeomFromWKT",
    "ST_AsText",
    "ST_AsGeoJSON",
    "Projection",
    "TransformAndProjection",
    "WktToWkb",
    "WkbToWkt",
]

import arctern
from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf("binary", PandasUDFType.SCALAR)
def Projection(geos, bottom_right, top_left, height, width):
    return arctern.projection(geos, bottom_right[0], top_left[0], height[0], width[0])

@pandas_udf("binary", PandasUDFType.SCALAR)
def TransformAndProjection(geos, src_rs, dst_rs, bottom_right, top_left, height, width):
    return arctern.transform_and_projection(geos, src_rs[0], dst_rs[0], bottom_right[0], top_left[0], height[0], width[0])

@pandas_udf("binary", PandasUDFType.SCALAR)
def WktToWkb(wkts):
    return arctern.wkt2wkb(wkts)

@pandas_udf("string", PandasUDFType.SCALAR)
def WkbToWkt(wkbs):
    return arctern.wkb2wkt(wkbs)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_PointFromText(geo):
    """
    Constructs point objects from the OGC Well-Known text representation.

    :type geo: pandas.Series.object
    :param geo: Geometries organized as WKT.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT (30 10)',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["data"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_PointFromText(data)) from data").show(100,0)
      +---------------------------------+
      |ST_AsText(ST_PointFromText(data))|
      +---------------------------------+
      |POINT (30 10)                    |
      +---------------------------------+
    """
    return arctern.ST_GeomFromText(geo)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_PolygonFromText(geo):
    """
    Constructs polygon objects from the OGC Well-Known text representation.

    :type geo: pandas.Series.object
    :param geo: Geometries organized as WKT.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["data"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_PolygonFromText(data)) from data").show(100,0)
      +-----------------------------------+
      |ST_AsText(ST_PolygonFromText(data))|
      +-----------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0))    |
      +-----------------------------------+
    """
    return arctern.ST_GeomFromText(geo)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_LineStringFromText(geo):
    """
    Constructs linestring objects from the OGC Well-Known text representation.

    :type geo: pandas.Series.object
    :param geo: Geometries organized as WKT.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING (0 0, 0 1, 1 1, 1 0)',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["data"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_LineStringFromText(data)) from data").show(100,0)
      +--------------------------------------+
      |ST_AsText(ST_LineStringFromText(data))|
      +--------------------------------------+
      |LINESTRING (0 0, 0 1, 1 1, 1 0)       |
      +--------------------------------------+
    """
    return arctern.ST_GeomFromText(geo)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_GeomFromWKT(geo):
    """
    Constructs geometry objects from the OGC Well-Known text representation.

    :type geo: pandas.Series.object
    :param geo: Geometries organized as WKT.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["data"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_GeomFromWKT(data)) from data").show(100,0)
      +-------------------------------+
      |ST_AsText(ST_GeomFromWKT(data))|
      +-------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0))|
      +-------------------------------+
    """
    return arctern.ST_GeomFromText(geo)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_GeomFromText(geo):
    """
    Constructs geometry objects from the OGC Well-Known text representation.

    :type geo: pandas.Series.object
    :param geo: Geometries organized as WKT.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["data"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_GeomFromText(data)) from data").show(100,0)
      +--------------------------------+
      |ST_AsText(ST_GeomFromText(data))|
      +--------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0)) |
      +--------------------------------+
    """
    return arctern.ST_GeomFromText(geo)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_AsText(geo):
    """
    Returns the Well-Known Text representation of the geometry.

    :type geo: pandas.Series.object
    :param geo: Geometries organized as WKB.

    :return: Geometries organized as WKT.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["data"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_GeomFromText(data)) from data").show(100,0)
      +--------------------------------+
      |ST_AsText(ST_GeomFromText(data))|
      +--------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0)) |
      +--------------------------------+
    """
    return arctern.ST_AsText(geo)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_AsGeoJSON(geo):
    """
    Returns the GeoJSON representation of the geometry.

    :type geo: pyarrow.array.string
    :param geo: Geometries organized as WKB.

    :return: Geometries organized as GeoJSON.
    :rtype: pyarrow.array.string

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["data"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsGeoJSON(ST_GeomFromText(data)) from data").show(100,0)
      +------------------------------------------------------------------------------------------------------------------+
      |ST_AsGeoJSON(ST_GeomFromText(data))                                                                               |
      +------------------------------------------------------------------------------------------------------------------+
      |{ "type": "Polygon", "coordinates": [ [ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ] ] ] }|
      +------------------------------------------------------------------------------------------------------------------+
    """
    return arctern.ST_AsGeoJSON(geo)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Point(x, y):
    """
    Construct Point geometries according to the coordinates.

    :type x: pandas.Series.float64
    :param x: Abscissa of the point.

    :type y: pandas.Series.float64
    :param y: Ordinate of the point.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> points_data = []
      >>> points_data.extend([(1,1)])
      >>> points_df = spark_session.createDataFrame(data=points_data, schema=["x", "y"]).cache()
      >>> points_df.createOrReplaceTempView("points")
      >>> spark_session.sql("select ST_AsText(ST_Point(x, y)) from points").show(100,0)
      +-------------------------+
      |ST_AsText(ST_Point(x, y))|
      +-------------------------+
      |POINT (1 1)              |
      +-------------------------+
    """
    return arctern.ST_Point(x, y)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_GeomFromGeoJSON(json):
    """
    Constructs a geometry object from the GeoJSON representation.

    :type json: pandas.Series.object
    :param json: Geometries organized as json

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([("{\"type\":\"Point\",\"coordinates\":[1,2]}",)])
      >>> test_data.extend([("{\"type\":\"LineString\",\"coordinates\":[[1,2],[4,5],[7,8]]}",)])
      >>> test_data.extend([("{\"type\":\"Polygon\",\"coordinates\":[[[0,0],[0,1],[1,1],[1,0],[0,0]]]}",)])
      >>> json_df = spark_session.createDataFrame(data=test_data, schema=["json"]).cache()
      >>> json_df.createOrReplaceTempView("json")
      >>> spark_session.sql("select ST_AsText(ST_GeomFromGeoJSON(json)) from json").show(100,0)
      +-----------------------------------+
      |ST_AsText(ST_GeomFromGeoJSON(json))|
      +-----------------------------------+
      |POINT (1 2)                        |
      +-----------------------------------+
      |LINESTRING (1 2,4 5,7 8)           |
      +-----------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0))    |
      +-----------------------------------+
    """
    return arctern.ST_GeomFromGeoJSON(json)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Intersection(left, right):
    """
    Calculate the point set intersection of geometries.

    For every (left, right) pair with the same offset value in left and right,
    calculate a geometry that represents their point set intersection.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT(0 0)', 'LINESTRING ( 2 0, 0 2 )')])
      >>> test_data.extend([('POINT(0 0)', 'LINESTRING ( 0 0, 2 2 )')])
      >>> intersection_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> intersection_df.createOrReplaceTempView("intersection")
      >>> spark_session.sql("select ST_AsText(ST_Intersection(ST_GeomFromText(left), ST_GeomFromText(right))) from intersection").show(100,0)
      +-------------------------------------------------------------------------+
      |ST_AsText(ST_Intersection(ST_GeomFromText(left), ST_GeomFromText(right)))|
      +-------------------------------------------------------------------------+
      |GEOMETRYCOLLECTION EMPTY                                                 |
      +-------------------------------------------------------------------------+
      |POINT (0 0)                                                              |
      +-------------------------------------------------------------------------+
    """
    return arctern.ST_Intersection(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsValid(geos):
    """
    For each item in geometries, check if it is of valid geometry format.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT (30 10)',)])
      >>> test_data.extend([('POINT (30 10)',)])
      >>> valid_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> valid_df.createOrReplaceTempView("valid")
      >>> spark_session.sql("select ST_IsValid(ST_GeomFromText(geos)) from valid").show(100,0)
      +---------------------------------+
      |ST_IsValid(ST_GeomFromText(geos))|
      +---------------------------------+
      |true                             |
      +---------------------------------+
      |true                             |
      +---------------------------------+
    """
    return arctern.ST_IsValid(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_PrecisionReduce(geos, precision):
    """
    Reduce the precision of geometry.

    For every geometry in geometries, reduce the decimal places of its coordinates
    to the given number. The last decimal place will be rounded.

    Note, the operation is performed NOT in "inplace" manner, i.e., new geometries
    in arrow::Array format will be construted and extra memory will be allocated.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKT.

    :type precision: uint32
    :param geos: The number to reduce the decimals places to.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT (10.777 11.888)',)])
      >>> precision_reduce_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> precision_reduce_df.createOrReplaceTempView("precision_reduce")
      >>> spark_session.sql("select ST_AsText(ST_PrecisionReduce(ST_GeomFromText(geos), 4)) from precision_reduce").show(100,0)
      +-------------------------------------------------------+
      |ST_AsText(ST_PrecisionReduce(ST_GeomFromText(geos), 4))|
      +-------------------------------------------------------+
      |POINT (10.78 11.89)                                    |
      +-------------------------------------------------------+
    """
    return arctern.ST_PrecisionReduce(geos, precision[0])

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Equals(left, right):
    """
    Check whether geometries are "spatially equal".

    For every (left, right) pair with the same offset value in left and right, check
    if they are "spatially equal". "Spatially equal" here means two geometries represent
    the same geometry structure.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(0 0, 10 10)', 'LINESTRING(0 0, 5 5, 10 10)')])
      >>> test_data.extend([('LINESTRING(10 10, 0 0)', 'LINESTRING(0 0, 5 5, 10 10)')])
      >>> equals_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> equals_df.createOrReplaceTempView("equals")
      >>> spark_session.sql("select ST_Equals(ST_GeomFromText(left), ST_GeomFromText(right)) from equals").show(100,0)
      +--------------------------------------------------------+
      |ST_Equals(ST_GeomFromText(left), ST_GeomFromText(right))|
      +--------------------------------------------------------+
      |true                                                    |
      +--------------------------------------------------------+
      |true                                                    |
      +--------------------------------------------------------+
    """
    return arctern.ST_Equals(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Touches(left, right):
    """
    Check whether geometries "touch".

    For every (left, right) pair with the same offset value in left and right, check
    if they "touch". "Touch" here means two geometries have common points, and the
    common points locate only on their boundaries.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(1 1)')])
      >>> test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(0 2)')])
      >>> touches_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> touches_df.createOrReplaceTempView("touches")
      >>> spark_session.sql("select ST_Touches(ST_GeomFromText(left), ST_GeomFromText(right)) from touches").show(100,0)
      +---------------------------------------------------------+
      |ST_Touches(ST_GeomFromText(left), ST_GeomFromText(right))|
      +---------------------------------------------------------+
      |false                                                    |
      +---------------------------------------------------------+
      |true                                                     |
      +---------------------------------------------------------+
    """
    return arctern.ST_Touches(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Overlaps(left, right):
    """
    Check whether geometries "spatially overlap".

    For every (left, right) pair with the same offset value in left and right, check
    if they "spatially overlap". "Spatially overlap" here means two geometries
    intersect but one does not completely contain another.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((1 1, 4 1, 4 5, 1 5, 1 1))', 'POLYGON((3 2, 6 2, 6 6, 3 6, 3 2))')])
      >>> test_data.extend([('POINT(1 0.5)', 'LINESTRING(1 0, 1 1, 3 5)')])
      >>> overlaps_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> overlaps_df.createOrReplaceTempView("overlaps")
      >>> spark.sql("select ST_Overlaps(ST_GeomFromText(left), ST_GeomFromText(right)) from overlaps").show(100,0)
      +----------------------------------------------------------+
      |ST_Overlaps(ST_GeomFromText(left), ST_GeomFromText(right))|
      +----------------------------------------------------------+
      |true                                                      |
      +----------------------------------------------------------+
      |false                                                     |
      +----------------------------------------------------------+
    """
    return arctern.ST_Overlaps(left, right)


@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Crosses(left, right):
    """
    Check whether geometries "spatially cross".

    For every (left, right) pair with the same offset value in left and right, check
    if they "spatially cross". "Spatially cross" here means two the geometries have
    some, but not all interior points in common. The intersection of the interiors of
    the geometries must not be the empty set and must have a dimensionality less than
    the maximum dimension of the two input geometries.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('MULTIPOINT((1 3), (4 1), (4 3))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
      >>> test_data.extend([('POLYGON((1 1, 4 1, 4 4, 1 4, 1 1))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
      >>> crosses_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> crosses_df.createOrReplaceTempView("crosses")
      >>> spark_session.sql("select ST_Crosses(ST_GeomFromText(left), ST_GeomFromText(right)) from crosses").show(100,0)
      +---------------------------------------------------------+
      |ST_Crosses(ST_GeomFromText(left), ST_GeomFromText(right))|
      +---------------------------------------------------------+
      |true                                                     |
      +---------------------------------------------------------+
      |false                                                    |
      +---------------------------------------------------------+
    """
    return arctern.ST_Crosses(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsSimple(geos):
    """
    Check whether geometry is "simple".

    For every geometry in geometries, check if it is "simple". "Simple" here means
    that a geometry has no anomalous geometric points such as self intersection or
    self tangency.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((1 2, 3 4, 5 6, 1 2))',)])
      >>> test_data.extend([('LINESTRING(1 1,2 2,2 3.5,1 3,1 2,2 1)',)])
      >>> simple_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> simple_df.createOrReplaceTempView("simple")
      >>> spark_session.sql("select ST_IsSimple(ST_GeomFromText(geos)) from simple").show(100,0)
      +----------------------------------+
      |ST_IsSimple(ST_GeomFromText(geos))|
      +----------------------------------+
      |false                             |
      +----------------------------------+
      |false                             |
      +----------------------------------+
    """
    return arctern.ST_IsSimple(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeometryType(geos):
    """
    For each geometry in geometries, return a string that indicates is type.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)',)])
      >>> test_data.extend([('POINT (30 10)',)])
      >>> geometry_type_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> geometry_type_df.createOrReplaceTempView("geometry_type")
      >>> spark_session.sql("select ST_GeometryType(ST_GeomFromText(geos)) from geometry_type").show(100,0)
      +--------------------------------------+
      |ST_GeometryType(ST_GeomFromText(geos))|
      +--------------------------------------+
      |ST_LINESTRING                         |
      +--------------------------------------+
      |POINT                                 |
      +--------------------------------------+
    """
    return arctern.ST_GeometryType(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_MakeValid(geos):
    """
    For every geometry in geometries, create a valid representation of it without
    losing any of the input vertices. Already-valid geometries won't have further
    intervention. This function returns geometries which are validated. Note, new
    geometries are construted in arrow::Array format, so extra memory will be allocated.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(0 0, 10 0, 20 0, 20 0, 30 0)',)])
      >>> test_data.extend([('POLYGON((1 5, 1 1, 3 3, 5 3, 7 1, 7 5, 5 3, 3 3, 1 5))',)])
      >>> make_valid_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> make_valid_df.createOrReplaceTempView("make_valid")
      >>> spark_session.sql("select ST_AsText(ST_MakeValid(ST_GeomFromText(geos))) from make_valid").show(100,0)
      +------------------------------------------------------------------------------------------------+
      |ST_AsText(ST_MakeValid(ST_GeomFromText(geos)))                                                  |
      +------------------------------------------------------------------------------------------------+
      |LINESTRING (0 0,10 0,20 0,20 0,30 0)                                                            |
      +------------------------------------------------------------------------------------------------+
      |GEOMETRYCOLLECTION (MULTIPOLYGON (((3 3,1 1,1 5,3 3)),((5 3,7 5,7 1,5 3))),LINESTRING (3 3,5 3))|
      +------------------------------------------------------------------------------------------------+
    """
    return arctern.ST_MakeValid(geos)

# TODO: ST_SimplifyPreserveTopology
@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    """
    For each geometry in geometries create a "simplified" version for it according
    to the precision that parameter tolerance specifies.

    Note simplified geometries with be construted in arrow::Array format, so extra
    memory will be allocated.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :type distance_tolerance: double
    :param distance_tolerance: The precision of the simplified geometry.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([(
          'POLYGON((8 25, 28 22, 28 20, 15 11, 33 3, 56 30, 46 33, 46 34, 47 44, 35 36, 45 33, 43 19, 29 21, 29 22, 35 26, 24 39, 8 25))',
          )])
      >>> test_data.extend([(
          'LINESTRING(250 250, 280 290, 300 230, 340 300, 360 260, 440 310, 470 360, 604 286)',
          )])
      >>> simplify_preserve_topology_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> simplify_preserve_topology_df.createOrReplaceTempView("simplify_preserve_topology")
      >>> spark_session.sql("select ST_AsText(ST_SimplifyPreserveTopology(ST_GeomFromText(geos), 10)) from simplify_preserve_topology").show(100,0)
      +----------------------------------------------------------------------------+
      |ST_AsText(ST_SimplifyPreserveTopology(ST_GeomFromText(geos), 10))           |
      +----------------------------------------------------------------------------+
      |POLYGON ((8 25,28 22,15 11,33 3,56 30,47 44,35 36,43 19,24 39,8 25))        |
      +----------------------------------------------------------------------------+
      |LINESTRING (250 250,280 290,300 230,340 300,360 260,440 310,470 360,604 286)|
      +----------------------------------------------------------------------------+
    """
    return arctern.ST_SimplifyPreserveTopology(geos, distance_tolerance[0])

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    """
    Construct polygon(rectangle) geometries from arr_min_x, arr_min_y, arr_max_x,
    arr_max_y. The edges of polygon are parallel to coordinate axis.

    :type min_x: pandas.Series.float64
    :param min_x: The x axis coordinates of the lower left vertical of the rectangles.

    :type min_y: pandas.Series.float64
    :param min_y: The y axis coordinates of the lower left vertical of the rectangles.

    :type max_x: pandas.Series.float64
    :param max_x: The x axis coordinates of the upper right vertical of the rectangles.

    :type max_y: pandas.Series.float64
    :param max_y: The y axis coordinates of the upper right vertical of the rectangles.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([(1.0, 3.0, 5.0, 7.0)])
      >>> test_data.extend([(2.0, 4.0, 6.0, 8.0)])
      >>> polygon_from_envelope_df = spark_session.createDataFrame(data=test_data, schema=['min_x', 'min_y', 'max_x', 'max_y']).cache()
      >>> polygon_from_envelope_df.createOrReplaceTempView('polygon_from_envelope')
      >>> spark_session.sql("select ST_AsText(ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y)) from polygon_from_envelope").show(100,0)
      +-------------------------------------------------------------+
      |ST_AsText(ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y))|
      +-------------------------------------------------------------+
      |POLYGON ((1 3,1 7,5 7,5 3,1 3))                              |
      +-------------------------------------------------------------+
      |POLYGON ((2 4,2 8,6 8,6 4,2 4))                              |
      +-------------------------------------------------------------+
    """
    return arctern.ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Contains(left, right):
    """
    Check whether a geometry contain another geometry.

    For every (left, right) pair with the same offset value in left and right, check
    if left_geometry "contains" right_geometry. Left "contains" right means no points
    of right_geometry lie in the exterior of left_geometry and at least one point of
    the interior of right_geometry lies in the interior of left_geometry.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((-1 3,2 1,0 -3,-1 3))','POLYGON((0 2,1 1,0 -1,0 2))')])
      >>> test_data.extend([('POLYGON((0 2,1 1,0 -1,0 2))','POLYGON((-1 3,2 1,0 -3,-1 3))')])
      >>> contains_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> contains_df.createOrReplaceTempView("contains")
      >>> spark_session.sql("select ST_Contains(ST_GeomFromText(left), ST_GeomFromText(right)) from contains").show(100,0)
      +----------------------------------------------------------+
      |ST_Contains(ST_GeomFromText(left), ST_GeomFromText(right))|
      +----------------------------------------------------------+
      |true                                                      |
      +----------------------------------------------------------+
      |false                                                     |
      +----------------------------------------------------------+
    """
    return arctern.ST_Contains(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Intersects(left, right):
    """
    Check whether two geometries intersect.

    For every (left, right) pair with the same offset value in left and right, check
    if left and right shares any portion of space.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT(0 0)', 'LINESTRING ( 0 0, 0 2 )')])
      >>> test_data.extend([('POINT(0 0)','LINESTRING ( 2 0, 0 2 )')])
      >>> intersects_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> intersects_df.createOrReplaceTempView("intersects")
      >>> spark_session.sql("select ST_Intersects(ST_GeomFromText(left), ST_GeomFromText(right)) from intersects").show(100,0)
      +------------------------------------------------------------+
      |ST_Intersects(ST_GeomFromText(left), ST_GeomFromText(right))|
      +------------------------------------------------------------+
      |true                                                        |
      +------------------------------------------------------------+
      |false                                                       |
      +------------------------------------------------------------+
    """
    return arctern.ST_Intersects(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Within(left, right):
    """
    Check whether a geometry is within another geometry.

    For every (left, right) pair with the same offset value in left and right, check
    if left is "within" right. Left "within" right means no points of left lie in the
    exterior of right and at least one point of the interior of left lies in the interior
    of right.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((2 2, 7 2, 7 5, 2 5, 2 2))','POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))')])
      >>> test_data.extend([('POLYGON((0 2, 5 2, 5 5, 0 5, 0 2))','POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))')])
      >>> within_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> within_df.createOrReplaceTempView("within")
      >>> spark_session.sql("select ST_Within(ST_GeomFromText(left), ST_GeomFromText(right)) from within").show(100,0)
      +--------------------------------------------------------+
      |ST_Within(ST_GeomFromText(left), ST_GeomFromText(right))|
      +--------------------------------------------------------+
      |true                                                    |
      +--------------------------------------------------------+
      |false                                                   |
      +--------------------------------------------------------+
    """
    return arctern.ST_Within(left, right)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Distance(left, right):
    """
    Calculate the distance between two geometries.

    For every (left, right) pair with the same offset value in left and right,
    calculates the minimum 2D Cartesian (planar) distance between left and right.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession .builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((-1 -1,2 2,0 1,-1 -1))','POLYGON((5 2,7 4,5 5,5 2))')])
      >>> test_data.extend([('POINT(31.75 31.25)','LINESTRING(32 32,32 35,40.5 35,32 35,32 32)')])
      >>> distance_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> distance_df.createOrReplaceTempView("distance")
      >>> spark_session.sql("select ST_Distance(ST_GeomFromText(left), ST_GeomFromText(right)) from distance").show(100,0)
      +----------------------------------------------------------+
      |ST_Distance(ST_GeomFromText(left), ST_GeomFromText(right))|
      +----------------------------------------------------------+
      |3                                                         |
      +----------------------------------------------------------+
      |0.7905694150420949                                        |
      +----------------------------------------------------------+
    """
    return arctern.ST_Distance(left, right)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_DistanceSphere(left, right):
    """
    Returns minimum distance in meters between two lon/lat points.
    Uses a spherical earth and radius derived from the spheroid defined by the SRID.

    For every (left, right) pair with the same offset value in left and right,
    calculates the minimum spherical distance between left and right.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
    TODO(dyh):: finish test
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession .builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT(31.75 31.25)','POINT(31.75 31.25)')])
      >>> test_data.extend([('POINT(31.75 31.25)','POINT(31.75 31.25)')])
      >>> distance_sphere_df = spark_session.createDataFrame(data=test_data, schema=["left", "right"]).cache()
      >>> distance_sphere_df.createOrReplaceTempView("distance_sphere")
      >>> spark_session.sql("select ST_Distance(ST_GeomFromText(left), ST_GeomFromText(right)) from distance_sphere").show(100,0)
      +----------------------------------------------------------+
      |ST_Distance(ST_GeomFromText(left), ST_GeomFromText(right))|
      +----------------------------------------------------------+
      |3                                                         |
      +----------------------------------------------------------+
      |0.7905694150420949                                        |
      +----------------------------------------------------------+
    """
    return arctern.ST_DistanceSphere(left, right)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Area(geos):
    """
    Calculate the area of geometry.

    For every geometry in geometries, calculate the 2D Cartesian (planar) area
    of geometry.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((10 20,10 30,20 30,30 10))',)])
      >>> test_data.extend([('POLYGON((10 20,10 40,30 40,40 10))',)])
      >>> area_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> area_df.createOrReplaceTempView("area")
      >>> spark_session.sql("select ST_Area(ST_GeomFromText(geos)) from area").show(100,0)
      +------------------------------+
      |ST_Area(ST_GeomFromText(geos))|
      +------------------------------+
      |200                           |
      +------------------------------+
      |600                           |
      +------------------------------+
    """
    return arctern.ST_Area(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Centroid(geos):
    """
    Compute the centroid of geometry.

    For every geometry in geometries, compute the controid point of geometry.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('MULTIPOINT ( -1 0, -1 2, -1 3, -1 4, -1 7, 0 1, 0 3, 1 1, 2 0, 6 0, 7 8, 9 8, 10 6 )',)])
      >>> test_data.extend([('CIRCULARSTRING(0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)',)])
      >>> centroid_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> centroid_df.createOrReplaceTempView("centroid")
      >>> spark_session.sql("select ST_AsText(ST_Centroid(ST_GeomFromText(geos))) from centroid").show(100,0)
      +---------------------------------------------+
      |ST_AsText(ST_Centroid(ST_GeomFromText(geos)))|
      +---------------------------------------------+
      |POINT (2.30769230769231 3.30769230769231)    |
      +---------------------------------------------+
      |POINT (0.5 1.0)                              |
      +---------------------------------------------+
    """
    return arctern.ST_Centroid(geos)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Length(geos):
    """
    Calculate the length of linear geometries.

    For every geometry in geometries, calculate the length of geometry.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(743238 2967416,743238 2967450,743265 2967450, 743265.625 2967416,743238 2967416)',)])
      >>> test_data.extend([('LINESTRING(-72.1260 42.45, -72.1240 42.45666, -72.123 42.1546)',)])
      >>> length_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> length_df.createOrReplaceTempView("length")
      >>> spark_session.sql("select ST_Length(ST_GeomFromText(geos)) from length").show(100,0)
      +--------------------------------+
      |ST_Length(ST_GeomFromText(geos))|
      +--------------------------------+
      |122.63074400009504              |
      +--------------------------------+
      |0.30901547439030225             |
      +--------------------------------+
    """
    return arctern.ST_Length(geos)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_HausdorffDistance(geo1, geo2):
    """
    Returns the Hausdorff distance between two geometries, a measure of how similar
    or dissimilar 2 geometries are.

    Implements algorithm for computing a distance metric which can be thought of as
    the "Discrete Hausdorff Distance". This is the Hausdorff distance restricted to
    discrete points for one of the geometries. Wikipedia article on Hausdorff distance

    Martin Davis note on how Hausdorff Distance calculation was used to prove
    correctness of the CascadePolygonUnion approach.

    When densifyFrac is specified, this function performs a segment densification before
    computing the discrete hausdorff distance. The densifyFrac parameter sets the fraction
    by which to densify each segment. Each segment will be split into a number of equal-length
    subsegments, whose fraction of the total length is closest to the given fraction.

    Units are in the units of the spatial reference system of the geometries.

    :type geo1: pandas.Series.object
    :param geo1: Geometries organized as WKB.

    :type geo2: pandas.Series.object
    :param geo2: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([("POLYGON((0 0 ,0 1, 1 1, 1 0, 0 0))", "POLYGON((0 0 ,0 2, 1 1, 1 0, 0 0))",)])
      >>> test_data.extend([("POINT(0 0)", "POINT(0 1)",)])
      >>> hausdorff_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> hausdorff_df.createOrReplaceTempView("hausdorff")
      >>> spark_session.sql("select ST_HausdorffDistance(ST_GeomFromText(geo1),ST_GeomFromText(geo2)) from hausdorff").show(100,0)
      +-----------------------------------------------------------------+
      |ST_HausdorffDistance(ST_GeomFromText(geo1),ST_GeomFromText(geo2))|
      +-----------------------------------------------------------------+
      |1                                                                |
      +-----------------------------------------------------------------+
      |1                                                                |
      +-----------------------------------------------------------------+
    """
    return arctern.ST_HausdorffDistance(geo1, geo2)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_ConvexHull(geos):
    """
    Compute the convex hull of geometry.

    Compute the smallest convex geometry that encloses all geometries for a geometry
    in geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('GEOMETRYCOLLECTION(POINT(1 1),POINT(0 0))',)])
      >>> test_data.extend([('GEOMETRYCOLLECTION(LINESTRING(2.5 3,-2 1.5), POLYGON((0 1,1 3,1 -2,0 1)))',)])
      >>> convexhull_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> convexhull_df.createOrReplaceTempView("convexhull")
      >>> spark_session.sql("select ST_AsText(ST_convexhull(ST_GeomFromText(geos))) from convexhull").show(100,0)
      +-----------------------------------------------+
      |ST_AsText(ST_convexhull(ST_GeomFromText(geos)))|
      +-----------------------------------------------+
      |LINESTRING (1 1,0 0)                           |
      +-----------------------------------------------+
      |POLYGON ((1 -2,-2.0 1.5,1 3,2.5 3.0,1 -2))     |
      +-----------------------------------------------+
    """
    return arctern.ST_ConvexHull(geos)

@pandas_udf("int", PandasUDFType.SCALAR)
def ST_NPoints(geos):
    """
    Calculates the points number for every geometry in geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return : An array of int64.
    :rtype : pandas.Series.int64

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)',)])
      >>> test_data.extend([('LINESTRING(77.29 29.07 1,77.42 29.26 0,77.27 29.31 -1,77.29 29.07 3)',)])
      >>> npoints_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> npoints_df.createOrReplaceTempView("npoints")
      >>> spark_session.sql("select ST_NPoints(ST_GeomFromText(geos)) from npoints").show(100,0)
      +---------------------------------+
      |ST_NPoints(ST_GeomFromText(geos))|
      +---------------------------------+
      |4                                |
      +---------------------------------+
      |4                                |
      +---------------------------------+
    """
    return arctern.ST_NPoints(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Envelope(geos):
    """
    Compute the double-precision minimum bounding box geometry for every geometry in geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('point (10 10)',)])
      >>> test_data.extend([('linestring (0 0 , 0 10)',)])
      >>> test_data.extend([('linestring (0 0 , 10 0)',)])
      >>> test_data.extend([('linestring (0 0 , 10 10)',)])
      >>> test_data.extend([('polygon ((0 0, 10 0, 10 10, 0 10, 0 0))',)])
      >>> test_data.extend([('multipoint (0 0, 10 0, 5 5)',)])
      >>> test_data.extend([('multilinestring ((0 0, 5 5), (6 6, 6 7, 10 10))',)])
      >>> test_data.extend([('multipolygon (((0 0, 10 0, 10 10, 0 10, 0 0), (11 11, 20 11, 20 20, 20 11, 11 11)))',)])
      >>> envelope_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> envelope_df.createOrReplaceTempView("envelope")
      >>> spark_session.sql("select ST_AsText(ST_Envelope(ST_GeomFromText(geos))) from envelope").show(100,0)
      +---------------------------------------------+
      |ST_AsText(ST_Envelope(ST_GeomFromText(geos)))|
      +---------------------------------------------+
      |POINT (10 10)                                |
      +---------------------------------------------+
      |LINESTRING (0 0,0 10)                        |
      +---------------------------------------------+
      |LINESTRING (0 0,10 0)                        |
      +---------------------------------------------+
      |POLYGON ((0 0,0 10,10 10,10 0,0 0))          |
      +---------------------------------------------+
      |POLYGON ((0 0,0 10,10 10,10 0,0 0))          |
      +---------------------------------------------+
      |POLYGON ((0 0,0 5,10 5,10 0,0 0))            |
      +---------------------------------------------+
      |POLYGON ((0 0,0 10,10 10,10 0,0 0))          |
      +---------------------------------------------+
      |POLYGON ((0 0,0 20,20 20,20 0,0 0))          |
      +---------------------------------------------+
    """
    return arctern.ST_Envelope(geos)

# TODO: ST_Buffer, how to polymorphicly define the behaviour of spark udf
@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Buffer(geos, dfDist):
    """
    Returns a geometry that represents all points whose distance from this geos is
    less than or equal to distance.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :type ofDist: int64
    :param ofDist: The maximum distance of the returned geometry from the given geometry.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT (0 1)',)])
      >>> buffer_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> buffer_df.createOrReplaceTempView("buffer")
      >>> spark_session.sql("select ST_AsText(ST_Buffer(ST_GeomFromText(geos), 0)) from buffer").show(100,0)
      +----------------------------------------------+
      |ST_AsText(ST_Buffer(ST_GeomFromText(geos), 0))|
      +----------------------------------------------+
      |POLYGON EMPTY                                 |
      +----------------------------------------------+
    """
    return arctern.ST_Buffer(geos, dfDist[0])

@pandas_udf("binary", PandasUDFType.GROUPED_AGG)
def ST_Union_Aggr(geos):
    """
    This function returns a MULTI geometry or NON-MULTI geometry from a set of geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometry organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data1 = []
      >>> test_data1.extend([('POLYGON ((1 1,1 2,2 2,2 1,1 1))',)])
      >>> test_data1.extend([('POLYGON ((2 1,3 1,3 2,2 2,2 1))',)])
      >>> union_aggr_df1 = spark_session.createDataFrame(data=test_data1, schema=['geos']).cache()
      >>> union_aggr_df1.createOrReplaceTempView("union_aggr1")
      >>> spark_session.sql("select ST_AsText(ST_Union_Aggr(ST_GeomFromText(geos))) from union_aggr1").show(100,0)
      +-----------------------------------------------+
      |ST_AsText(ST_Union_Aggr(ST_GeomFromText(geos)))|
      +-----------------------------------------------+
      |POLYGON ((4 1,4 0,0 0,0 4,4 4,4 2,5 2,5 1,4 1))|
      +-----------------------------------------------+
    """
    rst = arctern.ST_Union_Aggr(geos)
    return rst[0]

@pandas_udf("binary", PandasUDFType.GROUPED_AGG)
def ST_Envelope_Aggr(geos):
    """
    Compute the double-precision minimum bounding box geometry for every geometry in geometries,
    then returns a MULTI geometry or NON-MULTI geometry from a set of geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometry organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,4 0,4 4,0 4,0 0))',)])
      >>> test_data.extend([('POLYGON ((5 1,7 1,7 2,5 2,5 1))',)])
      >>> envelope_aggr_df = spark_session.createDataFrame(data=test_data, schema=['geos'])
      >>> envelope_aggr_df.createOrReplaceTempView('envelope_aggr')
      >>> spark_session.sql("select ST_AsText(ST_Envelope_Aggr(ST_GeomFromText(geos))) from envelope_aggr").show(100,0)
      +--------------------------------------------------+
      |ST_AsText(ST_Envelope_Aggr(ST_GeomFromText(geos)))|
      +--------------------------------------------------+
      |POLYGON ((0 0,0 4,7 4,7 0,0 0))                   |
      +--------------------------------------------------+
    """
    rst = arctern.ST_Envelope_Aggr(geos)
    return rst[0]

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Transform(geos, src_rs, dst_rs):
    """
    Returns a new geometry with its coordinates transformed to a different spatial
    reference system.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :type src_rs: string
    :param src_rs: The current srid of geometries.

    :type dst_rs: string
    :param dst_rs: The target srid of geometries tranfrom to.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>>  test_data = []
      >>> test_data.extend([('POINT (10 10)',)])
      >>> buffer_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> buffer_df.createOrReplaceTempView("buffer")
      >>> spark_session.sql("select ST_AsText(ST_Transform(ST_GeomFromText(geos), 'epsg:4326', 'epsg:3857')) from buffer").show(100,0)
      +------------------------------------------------------------------------+
      |ST_AsText(ST_Transform(ST_GeomFromText(geos), 'epsg:4326', 'epsg:3857'))|
      +------------------------------------------------------------------------+
      |POINT (1113194.90793274 1118889.97485796)                               |
      +------------------------------------------------------------------------+
    """
    return arctern.ST_Transform(geos, src_rs[0], dst_rs[0])

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_CurveToLine(geos):
    """
    Converts a CIRCULAR STRING to regular LINESTRING or CURVEPOLYGON to POLYGON or
    MULTISURFACE to MULTIPOLYGON. Useful for outputting to devices that can't support
    CIRCULARSTRING geometry types.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))',)])
      >>> buffer_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> buffer_df.createOrReplaceTempView("buffer")
      >>> rs=spark_session.sql("select ST_AsText(ST_CurveToLine(ST_GeomFromText(geos))) from buffer").collect()
      >>> assert str(rs[0][0]).startswith("POLYGON")
    """
    return arctern.ST_CurveToLine(geos)
