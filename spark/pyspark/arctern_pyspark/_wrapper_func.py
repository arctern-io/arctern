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
def Projection(geos, bottom_geo2, top_geo1, height, width):
    return arctern.projection(geos, bottom_geo2[0], top_geo1[0], height[0], width[0])

@pandas_udf("binary", PandasUDFType.SCALAR)
def TransformAndProjection(geos, src_rs, dst_rs, bottom_geo2, top_geo1, height, width):
    return arctern.transform_and_projection(geos, src_rs[0], dst_rs[0], bottom_geo2[0], top_geo1[0], height[0], width[0])

@pandas_udf("binary", PandasUDFType.SCALAR)
def WktToWkb(wkts):
    return arctern.wkt2wkb(wkts)

@pandas_udf("string", PandasUDFType.SCALAR)
def WkbToWkt(wkbs):
    return arctern.wkb2wkt(wkbs)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_PointFromText(geos):
    """
    Transform the representation of point from WKT to WKB.

    :type geos: WKT
    :param geos: Point in WKT form.

    :rtype: WKB
    :return: Point in WKB form.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT (30 10)',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_PointFromText(geos)) from data").show(100,0)
      +---------------------------------+
      |ST_AsText(ST_PointFromText(data))|
      +---------------------------------+
      |POINT (30 10)                    |
      +---------------------------------+
    """
    return arctern.ST_GeomFromText(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_PolygonFromText(geos):
    """
    Transform the representation of polygon from WKT to WKB.

    :type geos: WKT
    :param geos: Polygon in WKT form.

    :rtype: WKB
    :return: Polygon in WKB form.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_PolygonFromText(geos)) from data").show(100,0)
      +-----------------------------------+
      |ST_AsText(ST_PolygonFromText(data))|
      +-----------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0))    |
      +-----------------------------------+
    """
    return arctern.ST_GeomFromText(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_LineStringFromText(geos):
    """
    Transform the representation of linestring from WKT to WKB.

    :type geos: WKT
    :param geos: Linestring in WKT form.

    :rtype: WKB
    :return: Linestring in WKB form.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING (0 0, 0 1, 1 1, 1 0)',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_LineStringFromText(geos)) from data").show(100,0)
      +--------------------------------------+
      |ST_AsText(ST_LineStringFromText(data))|
      +--------------------------------------+
      |LINESTRING (0 0, 0 1, 1 1, 1 0)       |
      +--------------------------------------+
    """
    return arctern.ST_GeomFromText(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_GeomFromWKT(geos):
    """
    Transform the representation of geometry from WKT to WKB.

    :type geos: WKT
    :param geos: Geometry in WKT form.

    :rtype: WKB
    :return: Geometry in WKB form.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_GeomFromWKT(geos)) from data").show(100,0)
      +-------------------------------+
      |ST_AsText(ST_GeomFromWKT(data))|
      +-------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0))|
      +-------------------------------+
    """
    return arctern.ST_GeomFromText(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_GeomFromText(geos):
    """
    Transform the representation of geometry from WKT to WKB.

    :type geo: WKT
    :param geo: Geometry in WKT form.

    :rtype: WKB
    :return: Geometry in WKB form.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_GeomFromText(geos)) from data").show(100,0)
      +--------------------------------+
      |ST_AsText(ST_GeomFromText(geos))|
      +--------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0)) |
      +--------------------------------+
    """
    return arctern.ST_GeomFromText(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_AsText(geos):
    """
    Transform the representation of geometry from WKB to WKT.

    :type geo: WKB
    :param geo: Geometry in WKB form.

    :rtype: WKT
    :return: Geometry in WKT form.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsText(ST_GeomFromText(geos)) from data").show(100,0)
      +--------------------------------+
      |ST_AsText(ST_GeomFromText(geos))|
      +--------------------------------+
      |POLYGON ((0 0,0 1,1 1,1 0,0 0)) |
      +--------------------------------+
    """
    return arctern.ST_AsText(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_AsGeoJSON(geos):
    """
    Return the GeoJSON representation of the geometry.

    :type geo: WKB
    :param geo: Geometry in WKB form.

    :rtype: string
    :return: Geometry organized as GeoJSON format.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON ((0 0,0 1,1 1,1 0,0 0))',)])
      >>> data_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> data_df.createOrReplaceTempView("data")
      >>> spark_session.sql("select ST_AsGeoJSON(ST_GeomFromText(geos)) from data").show(100,0)
      +------------------------------------------------------------------------------------------------------------------+
      |ST_AsGeoJSON(ST_GeomFromText(geos))                                                                               |
      +------------------------------------------------------------------------------------------------------------------+
      |{ "type": "Polygon", "coordinates": [ [ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ] ] ] }|
      +------------------------------------------------------------------------------------------------------------------+
    """
    return arctern.ST_AsGeoJSON(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Point(x, y):
    """
    Construct point according to the coordinates.

    :type x: double
    :param x: Abscissa of the point.

    :type y: double
    :param y: Ordinate of the point.

    :rtype: WKB
    :return: Point in WKB form.

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
    Construct geometry from the GeoJSON representation.

    :type json: string
    :param json: Geometry in GeoJson format.

    :rtype: WKB
    :return: Geometry in WKB form.

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
      >>> json_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> json_df.createOrReplaceTempView("json")
      >>> spark_session.sql("select ST_AsText(ST_GeomFromGeoJSON(geos)) from json").show(100,0)
      +-----------------------------------+
      |ST_AsText(ST_GeomFromGeoJSON(geos))|
      +-----------------------------------+
      |POINT (1 2)                        |
      |LINESTRING (1 2,4 5,7 8)           |
      |POLYGON ((0 0,0 1,1 1,1 0,0 0))    |
      +-----------------------------------+
    """
    return arctern.ST_GeomFromGeoJSON(json)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Intersection(geo1, geo2):
    """
    Calculate the point set intersection of two geometry objects.

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: WKB
    :return: Geometry that represents the point set intersection.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT(0 0)', 'LINESTRING ( 2 0, 0 2 )')])
      >>> test_data.extend([('POINT(0 0)', 'LINESTRING ( 0 0, 2 2 )')])
      >>> intersection_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> intersection_df.createOrReplaceTempView("intersection")
      >>> spark_session.sql("select ST_AsText(ST_Intersection(ST_GeomFromText(geo1), ST_GeomFromText(geo2))) from intersection").show(100,0)
      +-------------------------------------------------------------------------+
      |ST_AsText(ST_Intersection(ST_GeomFromText(geo1), ST_GeomFromText(geo2)))|
      +-------------------------------------------------------------------------+
      |GEOMETRYCOLLECTION EMPTY                                                 |
      |POINT (0 0)                                                              |
      +-------------------------------------------------------------------------+
    """
    return arctern.ST_Intersection(geo1, geo2)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsValid(geos):
    """
    Check if geometry is of valid geometry format.

    :type geos: WKB
    :param geos: Geometry

    :rtype: boolean
    :return: True if geometry is valid.

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
      |true                             |
      +---------------------------------+
    """
    return arctern.ST_IsValid(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_PrecisionReduce(geos, precision):
    """
    For the coordinates of the geometry, reduce the number of significant digits
    to the given number. The last decimal place will be rounded.

    :type geos: WKB
    :param geos: Geometry

    :type precision: int
    :param precision: The number to of ignificant digits.

    :rtype: WKB
    :return: Geometry with reduced precision.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT (10.777 11.888)',)])
      >>> test_data.extend([('POLYGON ((10.23 11.25,6.0 0.0001,5.7854 8.56542,10.23 11.25))',)])
      >>> precision_reduce_df = spark_session.createDataFrame(data=test_data, schema=["geos"]).cache()
      >>> precision_reduce_df.createOrReplaceTempView("precision_reduce")
      >>> spark_session.sql("select ST_AsText(ST_PrecisionReduce(ST_GeomFromText(geos), 4)) from precision_reduce").show(100,0)
      +-------------------------------------------------------+
      |ST_AsText(ST_PrecisionReduce(ST_GeomFromText(geos), 4))|
      +-------------------------------------------------------+
      |POINT (10.78 11.89)                                    |
      |POLYGON ((10.23 11.25,6 0,5.785 8.565,10.23 11.25))    |
      +-------------------------------------------------------+
    """
    return arctern.ST_PrecisionReduce(geos, precision[0])

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Equals(geo1, geo2):
    """
    Check whether geometries are "spatially equal". "Spatially equal" here means two geometries represent
    the same geometry structure.

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: boolean
    :return: True if geometry "geo1" equals geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(0 0, 10 10)', 'LINESTRING(0 0, 5 5, 10 10)')])
      >>> test_data.extend([('LINESTRING(10 10, 0 0)', 'LINESTRING(0 0, 5 5, 10 10)')])
      >>> test_data.extend([('LINESTRING(0 0, 10 10)', 'LINESTRING(0 0, 5 5, 8 5)')])
      >>> equals_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> equals_df.createOrReplaceTempView("equals")
      >>> spark_session.sql("select ST_Equals(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from equals").show(100,0)
      +--------------------------------------------------------+
      |ST_Equals(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +--------------------------------------------------------+
      |true                                                    |
      |true                                                    |
      |false                                                   |
      +--------------------------------------------------------+
    """
    return arctern.ST_Equals(geo1, geo2)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Touches(geo1, geo2):
    """
    Check whether geometries "touch". "Touch" here means two geometries have common points, and the
    common points locate only on their boundaries.

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: boolean
    :return: True if geometry "geo1" touches geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(1 1)')])
      >>> test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(0 2)')])
      >>> touches_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> touches_df.createOrReplaceTempView("touches")
      >>> spark_session.sql("select ST_Touches(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from touches").show(100,0)
      +---------------------------------------------------------+
      |ST_Touches(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +---------------------------------------------------------+
      |false                                                    |
      |true                                                     |
      +---------------------------------------------------------+
    """
    return arctern.ST_Touches(geo1, geo2)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Overlaps(geo1, geo2):
    """
    Check whether geometries "spatially overlap". "Spatially overlap" here means two geometries
    intersect but one does not completely contain another.

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: boolean
    :return: True if geometry "geo1" overlap geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((1 1, 4 1, 4 5, 1 5, 1 1))', 'POLYGON((3 2, 6 2, 6 6, 3 6, 3 2))')])
      >>> test_data.extend([('POINT(1 0.5)', 'LINESTRING(1 0, 1 1, 3 5)')])
      >>> overlaps_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> overlaps_df.createOrReplaceTempView("overlaps")
      >>> spark.sql("select ST_Overlaps(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from overlaps").show(100,0)
      +----------------------------------------------------------+
      |ST_Overlaps(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +----------------------------------------------------------+
      |true                                                      |
      |false                                                     |
      +----------------------------------------------------------+
    """
    return arctern.ST_Overlaps(geo1, geo2)


@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Crosses(geo1, geo2):
    """
    Check whether geometries "spatially cross". "Spatially cross" here means two the geometries have
    some, but not all interior points in common. The intersection of the interiors of the geometries
    must not be the empty set and must have a dimensionality less than the maximum dimension of the two
    input geometries.

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: boolean
    :return: True if geometry "geo1" crosses geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('MULTIPOINT((1 3), (4 1), (4 3))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
      >>> test_data.extend([('POLYGON((1 1, 4 1, 4 4, 1 4, 1 1))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
      >>> crosses_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> crosses_df.createOrReplaceTempView("crosses")
      >>> spark_session.sql("select ST_Crosses(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from crosses").show(100,0)
      +---------------------------------------------------------+
      |ST_Crosses(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +---------------------------------------------------------+
      |true                                                     |
      |false                                                    |
      +---------------------------------------------------------+
    """
    return arctern.ST_Crosses(geo1, geo2)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsSimple(geos):
    """
    Check whether geometry is "simple". "Simple" here means that a geometry has no anomalous geometric points
    such as self intersection or self tangency.

    :type geos: WKB
    :param geos: Geometry

    :rtype: boolean
    :return: True if geometry is simple.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((1 2, 3 4, 5 6, 1 2))',)])
      >>> test_data.extend([('LINESTRING(1 1,2 2,2 3.5,1 3,1 2,2 1)',)])
      >>> test_data.extend([('POINT (1 1)')])
      >>> simple_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> simple_df.createOrReplaceTempView("simple")
      >>> spark_session.sql("select ST_IsSimple(ST_GeomFromText(geos)) from simple").show(100,0)
      +----------------------------------+
      |ST_IsSimple(ST_GeomFromText(geos))|
      +----------------------------------+
      |false                             |
      |false                             |
      |true                              |
      +----------------------------------+
    """
    return arctern.ST_IsSimple(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeometryType(geos):
    """
    For each geometry in geometries, return a string that indicates is type.

    :type geos: WKB
    :param geos: Geometry

    :rtype: string
    :return: The type of geometry, e.g., "ST_LINESTRING", "ST_POLYGON", "ST_POINT", "ST_MULTIPOINT".

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
      |ST_POINT                              |
      +--------------------------------------+
    """
    return arctern.ST_GeometryType(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_MakeValid(geos):
    """
    Create a valid representation of the geometry without losing any of the input vertices. If
    the geometry is already-valid, then nothing will be done.

    :type geos: WKB
    :param geos: Geometry

    :rtype: WKB
    :return: Geometry if the input geometry is already-valid or can be made valid. Otherwise, NULL.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('LINESTRING(0 0, 10 0, 20 0, 20 0, 30 0)',)])
      >>> test_data.extend([('POLYGON ((1 5, 1 1, 3 3, 5 3, 7 1, 7 5, 5 3, 3 3, 1 5))',)])
      >>> test_data.extend([('POLYGON ((1 1,1 5,3 3))',)])
      >>> make_valid_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> make_valid_df.createOrReplaceTempView("make_valid")
      >>> spark_session.sql("select ST_AsText(ST_MakeValid(ST_GeomFromText(geos))) from make_valid").show(100,0)
      +------------------------------------------------------------------------------------------------+
      |ST_AsText(ST_MakeValid(ST_GeomFromText(geos)))                                                  |
      +------------------------------------------------------------------------------------------------+
      |LINESTRING (0 0,10 0,20 0,20 0,30 0)                                                            |
      |GEOMETRYCOLLECTION (MULTIPOLYGON (((3 3,1 1,1 5,3 3)),((5 3,7 5,7 1,5 3))),LINESTRING (3 3,5 3))|
      |NULL                                                                                            |
      +------------------------------------------------------------------------------------------------+
    """
    return arctern.ST_MakeValid(geos)

# TODO: ST_SimplifyPreserveTopology
@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    """
    Returns a "simplified" version of the given geometry using the Douglas-Peucker algorithm.

    :type geos: WKB
    :param geos: Geometry

    :type distance_tolerance: double
    :param distance_tolerance: The maximum distance between a point on a linestring and a curve.

    :rtype: WKB
    :return: Geometry

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
      |LINESTRING (250 250,280 290,300 230,340 300,360 260,440 310,470 360,604 286)|
      +----------------------------------------------------------------------------+
    """
    return arctern.ST_SimplifyPreserveTopology(geos, distance_tolerance[0])

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    """
    Construct a polygon(rectangle) geometry from arr_min_x, arr_min_y, arr_max_x,
    arr_max_y. The edges of polygon are parallel to coordinate axis.

    :type min_x: double
    :param min_x: The minimum value of x coordinate of the rectangles.

    :type min_y: double
    :param min_y: The minimum value of y coordinate of the rectangles.

    :type max_x: double
    :param max_x: The maximum value of x coordinate of the rectangles.

    :type max_y: double
    :param max_y: The maximum value of y coordinate of the rectangles.

    :rtype: WKB
    :return: Geometry

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
      |POLYGON ((2 4,2 8,6 8,6 4,2 4))                              |
      +-------------------------------------------------------------+
    """
    return arctern.ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Contains(geo1, geo2):
    """
    Check whether geometry "geo1" contains geometry "geo2". "geo1 contains geo2" means no points
    of "geo2" lie in the exterior of "geo1" and at least one point of the interior of "geo2" lies
    in the interior of "geo1".

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: boolean
    :return: True if geometry "geo1" contains geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((-1 3,2 1,0 -3,-1 3))','POLYGON((0 2,1 1,0 -1,0 2))')])
      >>> test_data.extend([('POLYGON((0 2,1 1,0 -1,0 2))','POLYGON((-1 3,2 1,0 -3,-1 3))')])
      >>> contains_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> contains_df.createOrReplaceTempView("contains")
      >>> spark_session.sql("select ST_Contains(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from contains").show(100,0)
      +----------------------------------------------------------+
      |ST_Contains(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +----------------------------------------------------------+
      |true                                                      |
      |false                                                     |
      +----------------------------------------------------------+
    """
    return arctern.ST_Contains(geo1, geo2)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Intersects(geo1, geo2):
    """
    Check whether two geometries intersect (i.e., share any portion of space).

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: boolean
    :return: True if geometry "geo1" intersects geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT(0 0)', 'LINESTRING ( 0 0, 0 2 )')])
      >>> test_data.extend([('POINT(0 0)','LINESTRING ( 2 0, 0 2 )')])
      >>> intersects_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> intersects_df.createOrReplaceTempView("intersects")
      >>> spark_session.sql("select ST_Intersects(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from intersects").show(100,0)
      +------------------------------------------------------------+
      |ST_Intersects(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +------------------------------------------------------------+
      |true                                                        |
      |false                                                       |
      +------------------------------------------------------------+
    """
    return arctern.ST_Intersects(geo1, geo2)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Within(geo1, geo2):
    """
    Check whether geometry "geo1" is within geometry "geo2". "geo1 within geo2" means no points of "geo1" lie in the
    exterior of "geo2" and at least one point of the interior of "geo1" lies in the interior of "geo2".

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: boolean
    :return: True if geometry "geo1" within geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((2 2, 7 2, 7 5, 2 5, 2 2))','POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))')])
      >>> test_data.extend([('POLYGON((0 2, 5 2, 5 5, 0 5, 0 2))','POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))')])
      >>> within_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> within_df.createOrReplaceTempView("within")
      >>> spark_session.sql("select ST_Within(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from within").show(100,0)
      +--------------------------------------------------------+
      |ST_Within(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +--------------------------------------------------------+
      |true                                                    |
      |false                                                   |
      +--------------------------------------------------------+
    """
    return arctern.ST_Within(geo1, geo2)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Distance(geo1, geo2):
    """
    Calculates the minimum 2D Cartesian (planar) distance between "geo1" and "geo2".

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: double
    :return: The value that represents the distance between geometry "geo1" and geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession .builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POLYGON((-1 -1,2 2,0 1,-1 -1))','POLYGON((5 2,7 4,5 5,5 2))')])
      >>> test_data.extend([('POINT(31.75 31.25)','LINESTRING(32 32,32 35,40.5 35,32 35,32 32)')])
      >>> distance_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> distance_df.createOrReplaceTempView("distance")
      >>> spark_session.sql("select ST_Distance(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from distance").show(100,0)
      +----------------------------------------------------------+
      |ST_Distance(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +----------------------------------------------------------+
      |3                                                         |
      |0.7905694150420949                                        |
      +----------------------------------------------------------+
    """
    return arctern.ST_Distance(geo1, geo2)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_DistanceSphere(geo1, geo2):
    """
    Returns minimum distance in meters between two lon/lat points.Uses a spherical earth
    and radius derived from the spheroid defined by the SRID.

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: double
    :return: The value that represents the distance between geometry "geo1" and geometry "geo2".

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession .builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>> test_data = []
      >>> test_data.extend([('POINT(31.75 31.25)','POINT(31.75 31.25)')])
      >>> test_data.extend([('POINT(31.75 31.25)','POINT(31.75 31.25)')])
      >>> distance_sphere_df = spark_session.createDataFrame(data=test_data, schema=["geo1", "geo2"]).cache()
      >>> distance_sphere_df.createOrReplaceTempView("distance_sphere")
      >>> spark_session.sql("select ST_Distance(ST_GeomFromText(geo1), ST_GeomFromText(geo2)) from distance_sphere").show(100,0)
      +----------------------------------------------------------+
      |ST_Distance(ST_GeomFromText(geo1), ST_GeomFromText(geo2))|
      +----------------------------------------------------------+
      |3                                                         |
      |0.7905694150420949                                        |
      +----------------------------------------------------------+
    """
    return arctern.ST_DistanceSphere(geo1, geo2)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Area(geos):
    """
    Calculate the 2D Cartesian (planar) area of geometry.

    :type geos: WKB
    :param geos: Geometry

    :rtype: double
    :return: The value that represents the area of geometry.

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
      |600                           |
      +------------------------------+
    """
    return arctern.ST_Area(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Centroid(geos):
    """
    Compute the centroid of geometry.

    :type geos: WKB
    :param geos: Geometry

    :rtype: WKB
    :return: The centroid of geometry in WKB form.

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
      |POINT (0.5 1.0)                              |
      +---------------------------------------------+
    """
    return arctern.ST_Centroid(geos)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Length(geos):
    """
    Calculate the length of linear geometries.

    :type geos: WKB
    :param geos: Geometry

    :rtype: double
    :return: The value that represents the length of geometry.

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
      |0.30901547439030225             |
      +--------------------------------+
    """
    return arctern.ST_Length(geos)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_HausdorffDistance(geo1, geo2):
    """
    Returns the Hausdorff distance between two geometries, which is a measure of how similar
    two geometries are.

    :type geo1: WKB
    :param geo1: Geometry

    :type geo2: WKB
    :param geo2: Geometry

    :rtype: double
    :return: The value that represents the hausdorff distance between geo1 and geo2.

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
      |1                                                                |
      +-----------------------------------------------------------------+
    """
    return arctern.ST_HausdorffDistance(geo1, geo2)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_ConvexHull(geos):
    """
    Compute the smallest convex geometry that encloses all geometries in the given geometry.

    :type geos: WKB
    :param geos: Geometry

    :rtype: WKB
    :return: Geometry

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
      |POLYGON ((1 -2,-2.0 1.5,1 3,2.5 3.0,1 -2))     |
      +-----------------------------------------------+
    """
    return arctern.ST_ConvexHull(geos)

@pandas_udf("int", PandasUDFType.SCALAR)
def ST_NPoints(geos):
    """
    Calculate the number of points for the given geometry.

    :type geos: WKB
    :param geos: Geometry

    :rtype: int
    :return: The number of points.

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
      |4                                |
      +---------------------------------+
    """
    return arctern.ST_NPoints(geos)

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Envelope(geos):
    """
    Compute the double-precision minimum bounding box geometry for the given geometry.

    :type geos: WKB
    :param geos: Geometry

    :rtype: WKB
    :return: Geometry

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
      |LINESTRING (0 0,0 10)                        |
      |LINESTRING (0 0,10 0)                        |
      |POLYGON ((0 0,0 10,10 10,10 0,0 0))          |
      |POLYGON ((0 0,0 10,10 10,10 0,0 0))          |
      |POLYGON ((0 0,0 5,10 5,10 0,0 0))            |
      |POLYGON ((0 0,0 10,10 10,10 0,0 0))          |
      |POLYGON ((0 0,0 20,20 20,20 0,0 0))          |
      +---------------------------------------------+
    """
    return arctern.ST_Envelope(geos)

# TODO: ST_Buffer, how to polymorphicly define the behaviour of spark udf
@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_Buffer(geos, distance):
    """
    Return a geometry that represents all points whose distance from the given geometry is
    less than or equal to "distance".

    :type geos: WKB
    :param geos: Geometry

    :type distance: double
    :param distance: The maximum distance of the returned geometry from the given geometry.

    :rtype: WKB
    :return: Geometry

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
    return arctern.ST_Buffer(geos, distance[0])

@pandas_udf("binary", PandasUDFType.GROUPED_AGG)
def ST_Union_Aggr(geos):
    """
    Return a geometry that represents the union of a set of geometries.

    :type geos: WKB
    :param geos: Geometry

    :rtype: WKB
    :return: Geometry

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
    Compute the double-precision minimum bounding box geometry for the union of given geometries.

    :type geos: WKB
    :param geos: Geometry

    :rtype: WKB
    :return: Geometry

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
def ST_Transform(geos, from_srid, to_srid):
    """
    Return a new geometry with its coordinates transformed from spatial reference system "src_rs" to a "dst_rs".

    :type geos: WKB
    :param geos: Geometry

    :type from_srid: string
    :param from_srid: The current srid of geometries.

    :type to_srid: string
    :param to_srid: The target srid of geometries tranfrom to.

    :rtype: WKB
    :return: Geometry

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
    return arctern.ST_Transform(geos, from_srid[0], to_srid[0])

@pandas_udf("binary", PandasUDFType.SCALAR)
def ST_CurveToLine(geos):
    """
    Convert curves in a geometry to approximate linear representation, e,g., CIRCULAR STRING to regular LINESTRING, CURVEPOLYGON to POLYGON, and
    MULTISURFACE to MULTIPOLYGON. Useful for outputting to devices that can't support
    CIRCULARSTRING geometry types.

    :type geos: WKB
    :param geos: Geometry

    :rtype: WKB
    :return: Geometry

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
