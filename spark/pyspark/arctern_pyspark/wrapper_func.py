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

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PointFromText(geo):
    return arctern.ST_GeomFromText(geo)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PolygonFromText(geo):
    return arctern.ST_GeomFromText(geo)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_LineStringFromText(geo):
    return arctern.ST_GeomFromText(geo)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromWKT(geo):
    return arctern.ST_GeomFromText(geo)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromText(geo):
    return arctern.ST_GeomFromText(geo)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_AsText(geo):
    return geo

"""
**ST_Point**
*Introduction: Construct a Point from X and Y.*
example:
    points_data = []
    for i in range(2):
        points_data.extend([(i + 0.1, i + 0.1)])
    points_df = spark.createDataFrame(data=points_data, schema=["x", "y"]).cache()
    points_df.createOrReplaceTempView("points")
    spark.sql("select ST_Point(x, y) from points").show()

    result:
    ST_Point(x, y)
    --------------
    POINT (0.1 0.1)
    POINT (1.1 1.1)
    POINT (2.1 2.1)
"""
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Point(x, y):
    return arctern.ST_Point(x, y)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromGeoJSON(json):
    return arctern.ST_GeomFromGeoJSON(json)

"""
**ST_Intersection**
*Introduction: Return the intersection shape of two geometries. The return type is a geometry. The data in geometries is organized into WTK.*
example:
    test_data = []
    test_data.extend([('POINT(0 0)', 'LINESTRING ( 2 0, 0 2 )')])
    test_data.extend([('POINT(0 0)', 'LINESTRING ( 0 0, 2 2 )')])
    intersection_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    intersection_df.createOrReplaceTempView("intersection")
    spark.sql("select ST_Intersection(left, right) from intersection").show()
results:
    ST_Intersection(left,right)
    ---------------------------
     GEOMETRYCOLLECTION EMPTY
     POINT (0 0)   
"""
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Intersection(left, right):
    return arctern.ST_Intersection(left, right)

"""
**ST_IsValid**
*Introduction: Test if Geometry is valid.Return an array of boolean,geometries_1 is an array of WKT.*
example:
    test_data = []
    test_data.extend([('POINT (30 10)',)])
    test_data.extend([('POINT (30 10)',)])
    valid_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    valid_df.createOrReplaceTempView("valid")
    spark.sql("select ST_IsValid(geos) from valid").show()
results:
    ST_IsVaild(geos)
    ----------------
     true
     true
"""
@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsValid(geos):
    return arctern.ST_IsValid(geos)

"""
**ST_PrecisionReduce**
*Introduction: Reduce the precision of the given geometry to the given number of decimal places.Return an array of WTK,geometies_1 is an array of WKT.*
example:
    test_data = []
    test_data.extend([('POINT (10.777 11.888)',)])
    precision_reduce_df = spark.createDataFrame(data=test_data, schema=["geos"]).cache()
    precision_reduce_df.createOrReplaceTempView("precision_reduce")
    spark.sql("select ST_PrecisionReduce(geos, 4) from precision_reduce").show()
result:
    ST_PrecisionReduce(geos, 4)
    ---------------------------
    POINT (10.78 11.89)
"""
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PrecisionReduce(geos, precision):
    return arctern.ST_PrecisionReduce(geos, precision[0])

"""
**ST_Equals**
*Introduction: Test if leftGeometry is equal to rightGeometry.Return an array of boolean,The data in geometries is organized into WTK.*
example:
    test_data = []
    test_data.extend([('LINESTRING(0 0, 10 10)', 'LINESTRING(0 0, 5 5, 10 10)')])
    test_data.extend([('LINESTRING(10 10, 0 0)', 'LINESTRING(0 0, 5 5, 10 10)')])
    equals_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    equals_df.createOrReplaceTempView("equals")
    spark.sql("select ST_Equals(left, right) from equals").show()
results:
    ST_Equals(left, right)
    ----------------------
    true
    true
"""
@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Equals(left, right):
    return arctern.ST_Equals(left, right)

"""
**ST_Touches**
*Introduction: Test if leftGeometry touches rightGeometry.Return an array of boolean,The data in geometries is organized into WTK.*
example:
    test_data = []
    test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(1 1)')])
    test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(0 2)')])
    touches_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    touches_df.createOrReplaceTempView("touches")
    spark.sql("select ST_Touches(left, right) from touches").show()
results:
    ST_Touches(left, right)
    -----------------------
    false
    true
"""
@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Touches(left, right):
    return arctern.ST_Touches(left, right)

"""
**ST_Overlaps**
*Introduction: Test if leftGeometry overlaps rightGeometry.Return an array of boolean,The data in geometries is organized into WTK.*
example:
    test_data = []
    test_data.extend([('POLYGON((1 1, 4 1, 4 5, 1 5, 1 1))', 'POLYGON((3 2, 6 2, 6 6, 3 6, 3 2))')])
    test_data.extend([('POINT(1 0.5)', 'LINESTRING(1 0, 1 1, 3 5)')])
    overlaps_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    overlaps_df.createOrReplaceTempView("overlaps")
    spark.sql("select ST_Overlaps(left, right) from overlaps").show()
results:
    ST_Overlaps(left, right)
    ------------------------
    true
    false
"""
@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Overlaps(left, right):
    return arctern.ST_Overlaps(left, right)


"""
**ST_Crosses**
*Introduction: Test if leftGeometry crosses rightGeometry.Return an array of boolean,The data in geometries is organized into WTK.*
example:
    test_data = []
    test_data.extend([('MULTIPOINT((1 3), (4 1), (4 3))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
    test_data.extend([('POLYGON((1 1, 4 1, 4 4, 1 4, 1 1))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
    crosses_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    crosses_df.createOrReplaceTempView("crosses")
    spark.sql("select ST_Crosses(left, right) from crosses").show()
results:
    ST_Crosses(left, right)
    -----------------------
    true
    false
"""
@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Crosses(left, right):
    return arctern.ST_Crosses(left, right)

"""
**ST_IsSimple**
*Introduction: Test if Geometry is simple.Return an array of boolean,The data in geometries is organized into WTK.*
example:
    test_data = []
    test_data.extend([('POLYGON((1 2, 3 4, 5 6, 1 2))',)])
    test_data.extend([('LINESTRING(1 1,2 2,2 3.5,1 3,1 2,2 1)',)])
    simple_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    simple_df.createOrReplaceTempView("simple")
    spark.sql("select ST_IsSimple(geos) from simple").show()
results:
    ST_IsSimple(geos)
    -----------------
    false
    false
"""
@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsSimple(geos):
    return arctern.ST_IsSimple(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeometryType(geos):
    return arctern.ST_GeometryType(geos)

"""
**ST_MakeValid**
*Introduction: Given an invalid polygon or multipolygon and removeHoles boolean flag, create a valid representation of the geometry.*
example:
    test_data = []
    test_data.extend([('LINESTRING(0 0, 10 0, 20 0, 20 0, 30 0)',)])
    test_data.extend([('POLYGON((1 5, 1 1, 3 3, 5 3, 7 1, 7 5, 5 3, 3 3, 1 5))',)])
    make_valid_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    make_valid_df.createOrReplaceTempView("make_valid")
    spark.sql("select ST_MakeValid(geos) from make_valid").show()
results:
    ST_MakeValid(geos)
    ------------------
    LINESTRING (0 0,10 0,20 0,20 0,30 0)
    GEOMETRYCOLLECTION (MULTIPOLYGON (((3 3,1 1,1 5,3 3)),((5 3,7 5,7 1,5 3))),LINESTRING (3 3,5 3))
"""
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_MakeValid(geos):
    return arctern.ST_MakeValid(geos)

# TODO: ST_SimplifyPreserveTopology
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    return arctern.ST_SimplifyPreserveTopology(geos, distance_tolerance[0])

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    return arctern.ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Contains(left, right):
    return arctern.ST_Contains(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Intersects(left, right):
    return arctern.ST_Intersects(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Within(left, right):
    return arctern.ST_Within(left, right)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Distance(left, right):
    return arctern.ST_Distance(left, right)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Area(geos):
    return arctern.ST_Area(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Centroid(geos):
    return arctern.ST_Centroid(geos)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Length(geos):
    return arctern.ST_Length(geos)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_HausdorffDistance(geo1, geo2):
    return arctern.ST_HausdorffDistance(geo1, geo2)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_ConvexHull(geos):
    return arctern.ST_ConvexHull(geos)

@pandas_udf("int", PandasUDFType.SCALAR)
def ST_NPoints(geos):
    return arctern.ST_NPoints(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Envelope(geos):
    return arctern.ST_Envelope(geos)

# TODO: ST_Buffer, how to polymorphicly define the behaviour of spark udf
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Buffer(geos, dfDist):
    return arctern.ST_Buffer(geos, dfDist[0])

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def ST_Union_Aggr(geos):
    return arctern.ST_Union_Aggr(geos)

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def ST_Envelope_Aggr(geos):
    return arctern.ST_Envelope_Aggr(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Transform(geos, src_rs, dst_rs):
    return arctern.ST_Transform(geos, src_rs[0], dst_rs[0])

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_CurveToLine(geos):
    return arctern.ST_CurveToLine(geos)
