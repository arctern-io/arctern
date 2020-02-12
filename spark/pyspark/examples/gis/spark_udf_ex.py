from pyspark.sql import SparkSession
from zilliz_pyspark import register_funcs

def run_st_point(spark):
    points_data = []
    for i in range(10):
        points_data.extend([(i + 0.1, i + 0.1)])
    points_df = spark.createDataFrame(data = points_data, schema = ["x", "y"]).cache()
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
    intersection_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    intersection_df.createOrReplaceTempView("intersection")
    rs = spark.sql("select ST_Intersection_UDF(left, right) from intersection").collect()
    assert(rs[0][0] == 'POINT EMPTY')
    assert(rs[1][0] == 'POINT (0 0)')

def run_st_isvalid(spark):
    test_data = []
    test_data.extend([('POINT (30 10)')])
    test_data.extend([('i am not a valid geometry')])
    valid_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    valid_df.createOrReplaceTempView("valid")
    rs = spark.sql("select ST_IsValid_UDF(geos) from valid").collect()
    assert(rs[0][0])
    assert(not rs[1][0])

def run_st_equals(spark):
    test_data = []
    test_data.extend([('LINESTRING(0 0, 10 10)', 'LINESTRING(0 0, 5 5, 10 10)')])
    test_data.extend([('LINESTRING(10 10, 0 0)', 'LINESTRING(0 0, 5 5, 10 10)')])
    equals_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    equals_df.createOrReplaceTempView("equals")
    rs = spark.sql("select ST_Equals_UDF(left, right) from equals").collect()
    assert(not rs[0][0])
    assert(not rs[1][0])

def run_st_touches(spark):
    test_data = []
    test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(0 2)')])
    test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(1 1)')])
    touches_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    touches_df.createOrReplaceTempView("touches")
    rs = spark.sql("select ST_Touches_UDF(left, right) from touches").collect()
    assert(rs[1][0])
    assert(not rs[0][0])

def run_st_overlaps(spark):
    test_data = []
    test_data.extend([('POLYGON((1 1, 4 1, 4 5, 1 5, 1 1))', 'POLYGON((3 2, 6 2, 6 6, 3 6, 3 2))')])
    test_data.extend([('POINT(1 0.5)', 'LINESTRING(1 0, 1 1, 3 5)')])
    overlaps_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    overlaps_df.createOrReplaceTempView("overlaps")
    rs = spark.sql("select ST_Overlaps_UDF(left, right) from overlaps").collect()
    assert(rs[0][0])
    assert(not rs[1][0])

def run_st_crosses(spark):
    test_data = []
    test_data.extend([('MULTIPOINT((1 3), (4 1), (4 3))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
    test_data.extend([('POLYGON((1 1, 4 1, 4 4, 1 4, 1 1))', 'POLYGON((2 2, 5 2, 5 5, 2 5, 2 2))')])
    crosses_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    crosses_df.createOrReplaceTempView("crosses")
    rs = spark.sql("select ST_Crosses_UDF(left, right) from crosses").collect()
    assert(rs[0][0])
    assert(not rs[1][0])

def run_st_issimple(spark):
    test_data = []
    test_data.extend([('POLYGON((1 2, 3 4, 5 6, 1 2))')])
    test_data.extend([('LINESTRING(1 1,2 2,2 3.5,1 3,1 2,2 1)')])
    simple_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    simple_df.createOrReplaceTempView("simple")
    rs = spark.sql("select ST_IsSimple_UDF(geos) from simple").collect()
    assert(not rs[0][0])
    assert(not rs[1][0])

def run_st_geometry_type(spark):
    test_data = []
    test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)')])
    test_data.extend([('POINT (30 10)')])
    geometry_type_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    geometry_type_df.createOrReplaceTempView("geometry_type")
    rs = spark.sql("select ST_GeometryType_UDF(geos) from geometry_type").collect()
    assert(rs[0][0] == 'LINESTRING')
    assert(rs[1][0] == 'POINT')

def run_st_make_valid(spark):
    test_data = []
    test_data.extend([('LINESTRING(0 0, 10 0, 20 0, 20 0, 30 0)')])
    test_data.extend([('POLYGON((1 5, 1 1, 3 3, 5 3, 7 1, 7 5, 5 3, 3 3, 1 5))')])
    make_valid_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    make_valid_df.createOrReplaceTempView("make_valid")
    rs = spark.sql("select ST_MakeValid_UDF(geos) from make_valid").collect()
    assert(rs[0][0] == 'LINESTRING (0 0,10 0,20 0,20 0,30 0)')
    assert(rs[1][0] == 'GEOMETRYCOLLECTION (MULTIPOLYGON (((3 3,1 1,1 5,3 3)),((5 3,7 5,7 1,5 3))),LINESTRING (3 3,5 3))')

def run_st_simplify_preserve_topology(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((8 25, 28 22, 28 20, 15 11, 33 3, 56 30, 46 33, 46 34, 47 44, 35 36, 45 33, 43 19, 29 21, 29 22, 35 26, 24 39, 8 25))'
    )])
    test_data.extend([(
        'LINESTRING(250 250, 280 290, 300 230, 340 300, 360 260, 440 310, 470 360, 604 286)'
    )])
    simplify_preserve_topology_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    simplify_preserve_topology_df.createOrReplaceTempView("simplify_preserve_topology")
    rs = spark.sql("select ST_SimplifyPreserveTopology_UDF(geos) from simplify_preserve_topology").collect()
    assert(rs[0][0] == 'POLYGON ((8 25,28 22,15 11,33 3,56 30,47 44,35 36,43 19,24 39,8 25))')
    assert(rs[1][0] == 'LINESTRING (250 250,280 290,300 230,340 300,360 260,440 310,470 360,604 286)')

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
    rs = spark.sql("select ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y) from polygon_from_envelope").collect()
    assert(rs[0][0] == 'POLYGON ((1 3,5 3,1 7,5 7,1 3))')
    assert(rs[1][0] == 'POLYGON ((2 4,6 4,2 8,6 8,2 4))')

def run_st_contains(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((-1 3,2 1,0 -3,-1 3))'
        'POLYGON((0 2,1 1,0 -1,0 2))'
    )])
    test_data.extend([(
        'POLYGON((0 2,1 1,0 -1,0 2))',
        'POLYGON((-1 3,2 1,0 -3,-1 3))',
    )])
    contains_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    contains_df.createOrReplaceTempView("contains")
    rs = spark.sql("select ST_Contains_UDF(left, right) from contains").collect()
    assert(rs[0][0])
    assert(not rs[1][0])

def run_st_intersects(spark):
    test_data = []
    test_data.extend([(
        'POINT(0 0)',
        'LINESTRING ( 0 0, 0 2 )'
    )])
    test_data.extend([(
        'POINT(0 0)',
        'LINESTRING ( 2 0, 0 2 )',
    )])
    intersects_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    intersects_df.createOrReplaceTempView("intersects")
    rs = spark.sql("select ST_Intersects_UDF(left, right) from intersects").collect()
    assert(rs[0][0])
    assert(not rs[1][0])

def run_st_within(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((2 2, 7 2, 7 5, 2 5, 2 2))',
        'POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))',
    )])
    test_data.extend([(
        'POLYGON((0 2, 5 2, 5 5, 0 5, 0 2))'
        'POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))',
    )])
    within_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    within_df.createOrReplaceTempView("within")
    rs = spark.sql("select ST_Within_UDF(left, right) from within").collect()
    assert(rs[0][0])
    assert(not rs[1][0])

def run_st_distance(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((-1 -1,2 2,0 1,-1 -1))',
        'POLYGON((5 2,7 4,5 5,5 2))',
    )])
    test_data.extend([(
        'POINT(31.75 31.25)'
        'LINESTRING(32 32,32 35,40.5 35,32 35,32 32)'
    )])
    distance_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    distance_df.createOrReplaceTempView("distance")
    rs = spark.sql("select ST_Distance_UDF(left, right) from distance").collect()
    assert(rs[0][0] == 3)
    assert(rs[1][0] == 0.790569)

def run_st_area(spark):
    test_data = []
    test_data.extend([('POLYGON((10 20,10 30,20 30,30 10))')])
    test_data.extend([('POLYGON((10 20,10 40,30 40,40 10))')])
    area_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    area_df.createOrReplaceTempView("area")
    rs = spark.sql("select ST_Area_UDF(geos) from area").collect()
    assert(rs[0][0] == 200)
    assert(rs[1][0] == 600)

def run_st_centroid(spark):
    test_data = []
    test_data.extend([('MULTIPOINT ( -1 0, -1 2, -1 3, -1 4, -1 7, 0 1, 0 3, 1 1, 2 0, 6 0, 7 8, 9 8, 10 6 )')])
    test_data.extend([('CIRCULARSTRING(0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)')])
    centroid_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    centroid_df.createOrReplaceTempView("centroid")
    rs = spark.sql("select ST_Centroid_UDF(geos) from centroid").collect()
    assert(rs[0][0] == 'POINT (2.30769230769231 3.30769230769231)')
    assert(rs[1][0] == 'POINT (0.5 1.0)')

def run_st_length(spark):
    test_data = []
    test_data.extend([('LINESTRING(743238 2967416,743238 2967450,743265 2967450, 743265.625 2967416,743238 2967416)')])
    test_data.extend([('LINESTRING(-72.1260 42.45, -72.1240 42.45666, -72.123 42.1546)')])
    length_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    length_df.createOrReplaceTempView("length")
    rs = spark.sql("select ST_Length_UDF(geos) from length").collect()
    assert(rs[0][0] == 122.631)
    assert(rs[1][0] == 0.309015)

def run_st_npoints(spark):
    test_data = []
    test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)')])
    test_data.extend([('LINESTRING(77.29 29.07 1,77.42 29.26 0,77.27 29.31 -1,77.29 29.07 3)')])
    npoints_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    npoints_df.createOrReplaceTempView("npoints")
    rs = spark.sql("select ST_NPoints_UDF(geos) from npoints").collect()
    assert(rs[0][0] == 4)
    assert(rs[1][0] == 4)

def run_st_envelope(spark):
    test_data = []
    test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)')])
    test_data.extend([('LINESTRING(77.29 29.07 1,77.42 29.26 0,77.27 29.31 -1,77.29 29.07 3)')])
    envelope_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    envelope_df.createOrReplaceTempView("envelope")
    rs = spark.sql("select ST_Envelope_UDF(geos) from envelope").collect()
    assert(rs[0][0] == 'MULTIPOINT (0 0,1 3)')
    assert(rs[1][0] == 'LINESTRING (0 0,0 1,1.0000001 1.0,1.0000001 0.0,0 0)')

# TODO: ST_Buffer, find proper test case

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    register_funcs(spark)

    run_st_point(spark)

    spark.stop()
