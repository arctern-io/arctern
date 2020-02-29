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
from zilliz_pyspark import register_funcs

def run_st_point(spark):
    points_data = []
    for i in range(10):
        points_data.extend([(i + 0.1, i + 0.1)])
    points_df = spark.createDataFrame(data=points_data, schema=["x", "y"]).cache()
    points_df.createOrReplaceTempView("points")
    # rs has one column and ten row
    # rs[0] represent 1st row
    # rs[0][0] represent 1st column of 1st row
    rs = spark.sql("select ST_Point_UDF(x, y) from points").collect()
    for i in range(10):
        assert rs[i][0] == ('POINT (%.1f %.1f)' % (i + 0.1, i + 0.1))

def run_st_intersection(spark):
    test_data = []
    test_data.extend([('POINT(0 0)', 'LINESTRING ( 2 0, 0 2 )')])
    test_data.extend([('POINT(0 0)', 'LINESTRING ( 0 0, 2 2 )')])
    intersection_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    intersection_df.createOrReplaceTempView("intersection")
    rs = spark.sql("select ST_Intersection_UDF(left, right) from intersection").collect()
    assert rs[0][0] == 'POINT EMPTY'
    assert rs[1][0] == 'POINT (0 0)'

def run_st_isvalid(spark):
    test_data = []
    test_data.extend([('POINT (30 10)',)])
    test_data.extend([('POINT (30 10)',)])
    valid_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    valid_df.createOrReplaceTempView("valid")
    rs = spark.sql("select ST_IsValid_UDF(geos) from valid").collect()
    assert rs[0][0]
    assert rs[1][0]

def run_st_equals(spark):
    test_data = []
    test_data.extend([('LINESTRING(0 0, 10 10)', 'LINESTRING(0 0, 5 5, 10 10)')])
    test_data.extend([('LINESTRING(10 10, 0 0)', 'LINESTRING(0 0, 5 5, 10 10)')])
    equals_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    equals_df.createOrReplaceTempView("equals")
    rs = spark.sql("select ST_Equals_UDF(left, right) from equals").collect()
    assert not rs[0][0]
    assert not rs[1][0]

def run_st_touches(spark):
    test_data = []
    test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(1 1)')])
    test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(0 2)')])
    touches_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    touches_df.createOrReplaceTempView("touches")
    rs = spark.sql("select ST_Touches_UDF(left, right) from touches").collect()
    assert not rs[0][0]
    assert rs[1][0]

def run_st_overlaps(spark):
    test_data = []
    test_data.extend([('POLYGON((1 1, 4 1, 4 5, 1 5, 1 1))', 'POLYGON((3 2, 6 2, 6 6, 3 6, 3 2))')])
    test_data.extend([('POINT(1 0.5)', 'LINESTRING(1 0, 1 1, 3 5)')])
    overlaps_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    overlaps_df.createOrReplaceTempView("overlaps")
    rs = spark.sql("select ST_Overlaps_UDF(left, right) from overlaps").collect()
    assert rs[0][0]
    assert not rs[1][0]

def run_st_crosses(spark):
    test_data = []
    test_data.extend([('MULTIPOINT((1 3), (4 1), (4 3))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
    test_data.extend([('POLYGON((1 1, 4 1, 4 4, 1 4, 1 1))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
    crosses_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    crosses_df.createOrReplaceTempView("crosses")
    rs = spark.sql("select ST_Crosses_UDF(left, right) from crosses").collect()
    assert rs[0][0]
    assert not rs[1][0]

def run_st_issimple(spark):
    test_data = []
    test_data.extend([('POLYGON((1 2, 3 4, 5 6, 1 2))',)])
    test_data.extend([('LINESTRING(1 1,2 2,2 3.5,1 3,1 2,2 1)',)])
    simple_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    simple_df.createOrReplaceTempView("simple")
    rs = spark.sql("select ST_IsSimple_UDF(geos) from simple").collect()
    assert not rs[0][0]
    assert not rs[1][0]

def run_st_geometry_type(spark):
    test_data = []
    test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)',)])
    test_data.extend([('POINT (30 10)',)])
    geometry_type_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    geometry_type_df.createOrReplaceTempView("geometry_type")
    rs = spark.sql("select ST_GeometryType_UDF(geos) from geometry_type").collect()
    assert rs[0][0] == 'ST_LINESTRING'
    assert rs[1][0] == 'ST_POINT'

def run_st_make_valid(spark):
    test_data = []
    test_data.extend([('LINESTRING(0 0, 10 0, 20 0, 20 0, 30 0)',)])
    test_data.extend([('POLYGON((1 5, 1 1, 3 3, 5 3, 7 1, 7 5, 5 3, 3 3, 1 5))',)])
    make_valid_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    make_valid_df.createOrReplaceTempView("make_valid")
    rs = spark.sql("select ST_MakeValid_UDF(geos) from make_valid").collect()
    assert rs[0][0] == 'LINESTRING (0 0,10 0,20 0,20 0,30 0)'
    assert rs[1][0] == 'GEOMETRYCOLLECTION (MULTIPOLYGON (((3 3,1 1,1 5,3 3)),((5 3,7 5,7 1,5 3))),LINESTRING (3 3,5 3))'

def run_st_simplify_preserve_topology(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((8 25, 28 22, 28 20, 15 11, 33 3, 56 30, 46 33, 46 34, 47 44, 35 36, 45 33, 43 19, 29 21, 29 22, 35 26, 24 39, 8 25))',
    )])
    test_data.extend([(
        'LINESTRING(250 250, 280 290, 300 230, 340 300, 360 260, 440 310, 470 360, 604 286)',
    )])
    simplify_preserve_topology_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    simplify_preserve_topology_df.createOrReplaceTempView("simplify_preserve_topology")
    rs = spark.sql("select ST_SimplifyPreserveTopology_UDF(geos, 10) from simplify_preserve_topology").collect()
    assert rs[0][0] == 'POLYGON ((8 25,28 22,15 11,33 3,56 30,47 44,35 36,43 19,24 39,8 25))'
    assert rs[1][0] == 'LINESTRING (250 250,280 290,300 230,340 300,360 260,440 310,470 360,604 286)'

def run_st_polygon_from_envelope(spark):
    test_data = []
    test_data.extend([(
        1.0, 3.0, 5.0, 7.0
    )])
    test_data.extend([(
        2.0, 4.0, 6.0, 8.0
    )])
    polygon_from_envelope_df = spark.createDataFrame(data=test_data, schema=['min_x', 'min_y', 'max_x', 'max_y']).cache()
    polygon_from_envelope_df.createOrReplaceTempView('polygon_from_envelope')
    rs = spark.sql("select ST_PolygonFromEnvelope_UDF(min_x, min_y, max_x, max_y) from polygon_from_envelope").collect()
    assert rs[0][0] == 'POLYGON ((1 3,1 7,5 7,5 3,1 3))'
    assert rs[1][0] == 'POLYGON ((2 4,2 8,6 8,6 4,2 4))'

def run_st_contains(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((-1 3,2 1,0 -3,-1 3))',
        'POLYGON((0 2,1 1,0 -1,0 2))'
    )])
    test_data.extend([(
        'POLYGON((0 2,1 1,0 -1,0 2))',
        'POLYGON((-1 3,2 1,0 -3,-1 3))'
    )])
    contains_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    contains_df.createOrReplaceTempView("contains")
    rs = spark.sql("select ST_Contains_UDF(left, right) from contains").collect()
    assert rs[0][0]
    assert not rs[1][0]

def run_st_intersects(spark):
    test_data = []
    test_data.extend([(
        'POINT(0 0)',
        'LINESTRING ( 0 0, 0 2 )'
    )])
    test_data.extend([(
        'POINT(0 0)',
        'LINESTRING ( 2 0, 0 2 )'
    )])
    intersects_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    intersects_df.createOrReplaceTempView("intersects")
    rs = spark.sql("select ST_Intersects_UDF(left, right) from intersects").collect()
    assert rs[0][0]
    assert not rs[1][0]

def run_st_within(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((2 2, 7 2, 7 5, 2 5, 2 2))',
        'POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))'
    )])
    test_data.extend([(
        'POLYGON((0 2, 5 2, 5 5, 0 5, 0 2))',
        'POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))'
    )])
    within_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    within_df.createOrReplaceTempView("within")
    rs = spark.sql("select ST_Within_UDF(left, right) from within").collect()
    assert rs[0][0]
    assert not rs[1][0]

def run_st_distance(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((-1 -1,2 2,0 1,-1 -1))',
        'POLYGON((5 2,7 4,5 5,5 2))'
    )])
    test_data.extend([(
        'POINT(31.75 31.25)',
        'LINESTRING(32 32,32 35,40.5 35,32 35,32 32)'
    )])
    distance_df = spark.createDataFrame(data=test_data, schema=["left", "right"]).cache()
    distance_df.createOrReplaceTempView("distance")
    rs = spark.sql("select ST_Distance_UDF(left, right) from distance").collect()
    assert rs[0][0] == 3
    assert rs[1][0] == 0.7905694150420949

def run_st_area(spark):
    test_data = []
    test_data.extend([('POLYGON((10 20,10 30,20 30,30 10))',)])
    test_data.extend([('POLYGON((10 20,10 40,30 40,40 10))',)])
    area_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    area_df.createOrReplaceTempView("area")
    rs = spark.sql("select ST_Area_UDF(geos) from area").collect()
    assert rs[0][0] == 200
    assert rs[1][0] == 600

def run_st_centroid(spark):
    test_data = []
    test_data.extend([('MULTIPOINT ( -1 0, -1 2, -1 3, -1 4, -1 7, 0 1, 0 3, 1 1, 2 0, 6 0, 7 8, 9 8, 10 6 )',)])
    test_data.extend([('CIRCULARSTRING(0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)',)])
    centroid_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    centroid_df.createOrReplaceTempView("centroid")
    rs = spark.sql("select ST_Centroid_UDF(geos) from centroid").collect()
    assert rs[0][0] == 'POINT (2.30769230769231 3.30769230769231)'
    assert rs[1][0] == 'POINT (0.5 1.0)'

def run_st_length(spark):
    test_data = []
    test_data.extend([('LINESTRING(743238 2967416,743238 2967450,743265 2967450, 743265.625 2967416,743238 2967416)',)])
    test_data.extend([('LINESTRING(-72.1260 42.45, -72.1240 42.45666, -72.123 42.1546)',)])
    length_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    length_df.createOrReplaceTempView("length")
    rs = spark.sql("select ST_Length_UDF(geos) from length").collect()
    assert rs[0][0] == 122.63074400009504
    assert rs[1][0] == 0.30901547439030225

def run_st_convexhull(spark):
    test_data = []
    test_data.extend([('GEOMETRYCOLLECTION(POINT(1 1),POINT(0 0))',)])
    test_data.extend([('GEOMETRYCOLLECTION(LINESTRING(2.5 3,-2 1.5), POLYGON((0 1,1 3,1 -2,0 1)))',)])
    convexhull_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    convexhull_df.createOrReplaceTempView("convexhull")
    rs = spark.sql("select ST_convexhull_UDF(geos) from convexhull").collect()
    assert rs[0][0] == 'LINESTRING (1 1,0 0)'
    assert rs[1][0] == 'POLYGON ((1 -2,-2.0 1.5,1 3,2.5 3.0,1 -2))'

def run_st_npoints(spark):
    test_data = []
    test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)',)])
    test_data.extend([('LINESTRING(77.29 29.07 1,77.42 29.26 0,77.27 29.31 -1,77.29 29.07 3)',)])
    npoints_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    npoints_df.createOrReplaceTempView("npoints")
    rs = spark.sql("select ST_NPoints_UDF(geos) from npoints").collect()
    assert rs[0][0] == 4
    assert rs[1][0] == 4

def run_st_envelope(spark):
    test_data = []
    test_data.extend([('point (10 10)',)])
    test_data.extend([('linestring (0 0 , 0 10)',)])
    test_data.extend([('linestring (0 0 , 10 0)',)])
    test_data.extend([('linestring (0 0 , 10 10)',)])
    test_data.extend([('polygon ((0 0, 10 0, 10 10, 0 10, 0 0))',)])
    test_data.extend([('multipoint (0 0, 10 0, 5 5)',)])
    test_data.extend([('multilinestring ((0 0, 5 5), (6 6, 6 7, 10 10))',)])
    test_data.extend([('multipolygon (((0 0, 10 0, 10 10, 0 10, 0 0), (11 11, 20 11, 20 20, 20 11, 11 11)))',)])
    envelope_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    envelope_df.createOrReplaceTempView("envelope")
    rs = spark.sql("select ST_Envelope_UDF(geos) from envelope").collect()
    assert rs[0][0] == 'POINT (10 10)'
    assert rs[1][0] == 'LINESTRING (0 0,0 10)'
    assert rs[2][0] == 'LINESTRING (0 0,10 0)'
    assert rs[3][0] == 'POLYGON ((0 0,0 10,10 10,10 0,0 0))'
    assert rs[4][0] == 'POLYGON ((0 0,0 10,10 10,10 0,0 0))'
    assert rs[5][0] == 'POLYGON ((0 0,0 5,10 5,10 0,0 0))'
    assert rs[6][0] == 'POLYGON ((0 0,0 10,10 10,10 0,0 0))'
    assert rs[7][0] == 'POLYGON ((0 0,0 20,20 20,20 0,0 0))'

def run_st_buffer(spark):
    test_data = []
    test_data.extend([('POLYGON((0 0,1 0,1 1,0 0))',)])
    buffer_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    buffer_df.createOrReplaceTempView("buffer")
    rs = spark.sql("select ST_Buffer_UDF(geos, 1.2) from buffer").collect()
    assert rs[0][0] == 'POLYGON ((-0.848528137423857 0.848528137423857,0.151471862576143 1.84852813742386,0.19704327236937 1.89177379057287,0.244815530740195 1.93257515374836,0.294657697249032 1.97082039324994,0.346433157981967 2.00640468153451,0.4 2.03923048454133,0.455211400312543 2.06920782902604,0.511916028309039 2.09625454917112,0.569958460545639 2.12029651179664,0.629179606750062 2.14126781955418,0.689417145876974 2.15911099154688,0.750505971018688 2.17377712088057,0.812278641951722 2.18522600871417,0.874565844078815 2.19342627444193,0.937196852508467 2.19835544170549,1.0 2.2,1.06280314749153 2.19835544170549,1.12543415592118 2.19342627444193,1.18772135804828 2.18522600871417,1.24949402898131 2.17377712088057,1.31058285412302 2.15911099154688,1.37082039324994 2.14126781955418,1.43004153945436 2.12029651179664,1.48808397169096 2.09625454917112,1.54478859968746 2.06920782902604,1.6 2.03923048454133,1.65356684201803 2.00640468153451,1.70534230275097 1.97082039324994,1.75518446925981 1.93257515374836,1.80295672763063 1.89177379057287,1.84852813742386 1.84852813742386,1.89177379057287 1.80295672763063,1.93257515374837 1.7551844692598,1.97082039324994 1.70534230275097,2.00640468153451 1.65356684201803,2.03923048454133 1.6,2.06920782902604 1.54478859968746,2.09625454917112 1.48808397169096,2.12029651179664 1.43004153945436,2.14126781955418 1.37082039324994,2.15911099154688 1.31058285412302,2.17377712088057 1.24949402898131,2.18522600871417 1.18772135804828,2.19342627444193 1.12543415592118,2.19835544170549 1.06280314749153,2.2 1.0,2.2 0.0,2.19835544170549 -0.062803147491532,2.19342627444193 -0.125434155921184,2.18522600871417 -0.187721358048277,2.17377712088057 -0.249494028981311,2.15911099154688 -0.310582854123025,2.14126781955418 -0.370820393249937,2.12029651179664 -0.43004153945436,2.09625454917112 -0.48808397169096,2.06920782902604 -0.544788599687456,2.03923048454133 -0.6,2.00640468153451 -0.653566842018033,1.97082039324994 -0.705342302750968,1.93257515374836 -0.755184469259805,1.89177379057287 -0.80295672763063,1.84852813742386 -0.848528137423857,1.80295672763063 -0.891773790572873,1.75518446925981 -0.932575153748365,1.70534230275097 -0.970820393249937,1.65356684201803 -1.00640468153451,1.6 -1.03923048454133,1.54478859968746 -1.06920782902604,1.48808397169096 -1.09625454917112,1.43004153945436 -1.12029651179664,1.37082039324994 -1.14126781955418,1.31058285412302 -1.15911099154688,1.24949402898131 -1.17377712088057,1.18772135804828 -1.18522600871417,1.12543415592118 -1.19342627444193,1.06280314749153 -1.19835544170549,1.0 -1.2,0.0 -1.2,-0.062803147491532 -1.19835544170549,-0.125434155921184 -1.19342627444193,-0.187721358048276 -1.18522600871417,-0.24949402898131 -1.17377712088057,-0.310582854123024 -1.15911099154688,-0.370820393249936 -1.14126781955418,-0.430041539454359 -1.12029651179664,-0.488083971690959 -1.09625454917112,-0.544788599687455 -1.06920782902604,-0.6 -1.03923048454133,-0.653566842018031 -1.00640468153451,-0.705342302750966 -0.970820393249938,-0.755184469259804 -0.932575153748366,-0.802956727630628 -0.891773790572875,-0.848528137423855 -0.848528137423859,-0.891773790572871 -0.802956727630632,-0.932575153748363 -0.755184469259807,-0.970820393249935 -0.70534230275097,-1.00640468153451 -0.653566842018035,-1.03923048454132 -0.6,-1.06920782902604 -0.544788599687459,-1.09625454917112 -0.488083971690964,-1.12029651179664 -0.430041539454364,-1.14126781955418 -0.370820393249941,-1.15911099154688 -0.310582854123029,-1.17377712088057 -0.249494028981315,-1.18522600871416 -0.187721358048281,-1.19342627444193 -0.125434155921189,-1.19835544170549 -0.062803147491537,-1.2 -0.0,-1.19835544170549 0.062803147491527,-1.19342627444193 0.125434155921179,-1.18522600871417 0.187721358048272,-1.17377712088057 0.249494028981306,-1.15911099154688 0.310582854123019,-1.14126781955419 0.370820393249931,-1.12029651179664 0.430041539454355,-1.09625454917112 0.488083971690954,-1.06920782902604 0.54478859968745,-1.03923048454133 0.6,-1.00640468153451 0.653566842018027,-0.970820393249941 0.705342302750962,-0.93257515374837 0.755184469259799,-0.891773790572878 0.802956727630624,-0.848528137423857 0.848528137423857))'

def run_st_union_aggr(spark):
    test_data1 = []
    test_data1.extend([('POLYGON ((1 1,1 2,2 2,2 1,1 1))',)])
    test_data1.extend([('POLYGON ((2 1,3 1,3 2,2 2,2 1))',)])
    union_aggr_df1 = spark.createDataFrame(data=test_data1, schema=['geos']).cache()
    union_aggr_df1.createOrReplaceTempView("union_aggr1")
    rs = spark.sql("select ST_Union_Aggr_UDF(geos) from union_aggr1").collect()
    assert rs[0][0] == 'POLYGON ((1 1,1 2,2 2,3 2,3 1,2 1,1 1))'

    test_data2 = []
    test_data2.extend([('POLYGON ((0 0,4 0,4 4,0 4,0 0))',)])
    test_data2.extend([('POLYGON ((3 1,5 1,5 2,3 2,3 1))',)])
    union_aggr_df2 = spark.createDataFrame(data=test_data2, schema=['geos']).cache()
    union_aggr_df2.createOrReplaceTempView("union_aggr2")
    rs = spark.sql("select ST_Union_Aggr_UDF(geos) from union_aggr2").collect()
    assert rs[0][0] == 'POLYGON ((4 1,4 0,0 0,0 4,4 4,4 2,5 2,5 1,4 1))'

    test_data3 = []
    test_data3.extend([('POLYGON ((0 0,4 0,4 4,0 4,0 0))',)])
    test_data3.extend([('POLYGON ((5 1,7 1,7 2,5 2,5 1))',)])
    union_aggr_df3 = spark.createDataFrame(data=test_data3, schema=['geos']).cache()
    union_aggr_df3.createOrReplaceTempView("union_aggr3")
    rs = spark.sql("select ST_Union_Aggr_UDF(geos) from union_aggr3").collect()
    assert rs[0][0] == 'MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)),((5 1,7 1,7 2,5 2,5 1)))'

    test_data4 = []
    test_data4.extend([('POLYGON ((0 0,0 4,4 4,4 0,0 0))',)])
    test_data4.extend([('POINT (2 3)',)])
    union_aggr_df4 = spark.createDataFrame(data=test_data4, schema=['geos']).cache()
    union_aggr_df4.createOrReplaceTempView("union_aggr4")
    rs = spark.sql("select ST_Union_Aggr_UDF(geos) from union_aggr4").collect()
    assert rs[0][0] == 'POLYGON ((0 0,0 4,4 4,4 0,0 0))'

def run_st_envelope_aggr(spark):
    test_data = []
    test_data.extend([('POLYGON ((0 0,4 0,4 4,0 4,0 0))',)])
    test_data.extend([('POLYGON ((5 1,7 1,7 2,5 2,5 1))',)])
    envelope_aggr_df = spark.createDataFrame(data=test_data, schema=['geos'])
    envelope_aggr_df.createOrReplaceTempView('envelope_aggr')
    rs = spark.sql("select ST_Envelope_Aggr_UDF(geos) from envelope_aggr").collect()
    assert rs[0][0] == 'POLYGON ((0 0,0 4,7 4,7 0,0 0))'

def run_st_transform(spark):
    test_data = []
    test_data.extend([('POINT (10 10)',)])
    buffer_df = spark.createDataFrame(data=test_data, schema=['geos']).cache()
    buffer_df.createOrReplaceTempView("buffer")
    rs = spark.sql("select ST_Transform_UDF(geos, 'epsg:4326', 'epsg:3857') from buffer").collect()
    assert rs[0][0] == 'POINT (1113194.90793274 1118889.97485796)'


if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    register_funcs(spark_session)

    run_st_point(spark_session)
    run_st_intersection(spark_session)
    run_st_isvalid(spark_session)
    run_st_equals(spark_session)
    run_st_touches(spark_session)
    run_st_overlaps(spark_session)
    run_st_crosses(spark_session)
    run_st_issimple(spark_session)
    run_st_geometry_type(spark_session)
    run_st_make_valid(spark_session)
    run_st_simplify_preserve_topology(spark_session)
    run_st_polygon_from_envelope(spark_session)
    run_st_contains(spark_session)
    run_st_intersects(spark_session)
    run_st_within(spark_session)
    run_st_distance(spark_session)
    run_st_area(spark_session)
    run_st_centroid(spark_session)
    run_st_length(spark_session)
    run_st_convexhull(spark_session)
    run_st_npoints(spark_session)
    run_st_envelope(spark_session)
    run_st_buffer(spark_session)
    run_st_union_aggr(spark_session)
    run_st_envelope_aggr(spark_session)
    run_st_transform(spark_session)

    spark_session.stop()
