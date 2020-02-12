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
    test_data.extend([('POINT (30 10)',)])
    test_data.extend([('POINT (30 10)',)])
    valid_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    valid_df.createOrReplaceTempView("valid")
    rs = spark.sql("select ST_IsValid_UDF(geos) from valid").collect()
    assert(rs[0][0])
    assert(rs[1][0])

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
    test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(1 1)')])
    test_data.extend([('LINESTRING(0 0, 1 1, 0 2)', 'POINT(0 2)')])
    touches_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    touches_df.createOrReplaceTempView("touches")
    rs = spark.sql("select ST_Touches_UDF(left, right) from touches").collect()
    assert(not rs[0][0])
    assert(rs[1][0])

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
    test_data.extend([('POLYGON((1 2, 3 4, 5 6, 1 2))',)])
    test_data.extend([('LINESTRING(1 1,2 2,2 3.5,1 3,1 2,2 1)',)])
    simple_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    simple_df.createOrReplaceTempView("simple")
    rs = spark.sql("select ST_IsSimple_UDF(geos) from simple").collect()
    assert(not rs[0][0])
    assert(not rs[1][0])

def run_st_geometry_type(spark):
    test_data = []
    test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)',)])
    test_data.extend([('POINT (30 10)',)])
    geometry_type_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    geometry_type_df.createOrReplaceTempView("geometry_type")
    rs = spark.sql("select ST_GeometryType_UDF(geos) from geometry_type").collect()
    assert(rs[0][0] == 'LINESTRING')
    assert(rs[1][0] == 'POINT')

def run_st_make_valid(spark):
    test_data = []
    test_data.extend([('LINESTRING(0 0, 10 0, 20 0, 20 0, 30 0)',)])
    test_data.extend([('POLYGON((1 5, 1 1, 3 3, 5 3, 7 1, 7 5, 5 3, 3 3, 1 5))',)])
    make_valid_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    make_valid_df.createOrReplaceTempView("make_valid")
    rs = spark.sql("select ST_MakeValid_UDF(geos) from make_valid").collect()
    assert(rs[0][0] == 'LINESTRING (0 0,10 0,20 0,20 0,30 0)')
    assert(rs[1][0] == 'GEOMETRYCOLLECTION (MULTIPOLYGON (((3 3,1 1,1 5,3 3)),((5 3,7 5,7 1,5 3))),LINESTRING (3 3,5 3))')

def run_st_simplify_preserve_topology(spark):
    test_data = []
    test_data.extend([(
        'POLYGON((8 25, 28 22, 28 20, 15 11, 33 3, 56 30, 46 33, 46 34, 47 44, 35 36, 45 33, 43 19, 29 21, 29 22, 35 26, 24 39, 8 25))',
    )])
    test_data.extend([(
        'LINESTRING(250 250, 280 290, 300 230, 340 300, 360 260, 440 310, 470 360, 604 286)',
    )])
    simplify_preserve_topology_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    simplify_preserve_topology_df.createOrReplaceTempView("simplify_preserve_topology")
    rs = spark.sql("select ST_SimplifyPreserveTopology_UDF(geos, 10) from simplify_preserve_topology").collect()
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
    rs = spark.sql("select ST_PolygonFromEnvelope_UDF(min_x, min_y, max_x, max_y) from polygon_from_envelope").collect()
    assert(rs[0][0] == 'POLYGON ((1 3,5 3,1 7,5 7,1 3))')
    assert(rs[1][0] == 'POLYGON ((2 4,6 4,2 8,6 8,2 4))')

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
        'LINESTRING ( 2 0, 0 2 )'
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
        'POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))'
    )])
    test_data.extend([(
        'POLYGON((0 2, 5 2, 5 5, 0 5, 0 2))',
        'POLYGON((1 1, 8 1, 8 7, 1 7, 1 1))'
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
        'POLYGON((5 2,7 4,5 5,5 2))'
    )])
    test_data.extend([(
        'POINT(31.75 31.25)',
        'LINESTRING(32 32,32 35,40.5 35,32 35,32 32)'
    )])
    distance_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    distance_df.createOrReplaceTempView("distance")
    rs = spark.sql("select ST_Distance_UDF(left, right) from distance").collect()
    assert(rs[0][0] == 3)
    assert(rs[1][0] == 0.7905694150420949)

def run_st_area(spark):
    test_data = []
    test_data.extend([('POLYGON((10 20,10 30,20 30,30 10))',)])
    test_data.extend([('POLYGON((10 20,10 40,30 40,40 10))',)])
    area_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    area_df.createOrReplaceTempView("area")
    rs = spark.sql("select ST_Area_UDF(geos) from area").collect()
    assert(rs[0][0] == 200)
    assert(rs[1][0] == 600)

def run_st_centroid(spark):
    test_data = []
    test_data.extend([('MULTIPOINT ( -1 0, -1 2, -1 3, -1 4, -1 7, 0 1, 0 3, 1 1, 2 0, 6 0, 7 8, 9 8, 10 6 )',)])
    test_data.extend([('CIRCULARSTRING(0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)',)])
    centroid_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    centroid_df.createOrReplaceTempView("centroid")
    rs = spark.sql("select ST_Centroid_UDF(geos) from centroid").collect()
    assert(rs[0][0] == 'POINT (2.30769230769231 3.30769230769231)')
    assert(rs[1][0] == 'POINT (0.5 1.0)')

def run_st_length(spark):
    test_data = []
    test_data.extend([('LINESTRING(743238 2967416,743238 2967450,743265 2967450, 743265.625 2967416,743238 2967416)',)])
    test_data.extend([('LINESTRING(-72.1260 42.45, -72.1240 42.45666, -72.123 42.1546)',)])
    length_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    length_df.createOrReplaceTempView("length")
    rs = spark.sql("select ST_Length_UDF(geos) from length").collect()
    assert(rs[0][0] == 122.63074400009504)
    assert(rs[1][0] == 0.30901547439030225)

def run_st_convexhull(spark):
    test_data = []
    test_data.extend([('GEOMETRYCOLLECTION(POINT(1 1),POINT(0 0))',)])
    test_data.extend([('GEOMETRYCOLLECTION(LINESTRING(2.5 3,-2 1.5), POLYGON((0 1,1 3,1 -2,0 1)))',)])
    convexhull_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    convexhull_df.createOrReplaceTempView("convexhull")
    rs = spark.sql("select ST_convexhull_UDF(geos) from convexhull").collect()
    assert(rs[0][0] == 'LINESTRING (1 1,0 0)')
    assert(rs[1][0] == 'POLYGON ((1 -2,-2.0 1.5,1 3,2.5 3.0,1 -2))')

def run_st_npoints(spark):
    test_data = []
    test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)',)])
    test_data.extend([('LINESTRING(77.29 29.07 1,77.42 29.26 0,77.27 29.31 -1,77.29 29.07 3)',)])
    npoints_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    npoints_df.createOrReplaceTempView("npoints")
    rs = spark.sql("select ST_NPoints_UDF(geos) from npoints").collect()
    assert(rs[0][0] == 4)
    assert(rs[1][0] == 4)

def run_st_envelope(spark):
    test_data = []
    test_data.extend([('LINESTRING(77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)',)])
    test_data.extend([('LINESTRING(77.29 29.07 1,77.42 29.26 0,77.27 29.31 -1,77.29 29.07 3)',)])
    envelope_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    envelope_df.createOrReplaceTempView("envelope")
    rs = spark.sql("select ST_Envelope_UDF(geos) from envelope").collect()
    assert(rs[0][0] == 'MULTIPOINT EMPTY')
    assert(rs[1][0] == 'MULTIPOINT EMPTY')

def run_st_buffer(spark):
    test_data = []
    test_data.extend([('POLYGON((0 0,1 0,1 1,0 0))',)])
    buffer_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    buffer_df.createOrReplaceTempView("buffer")
    rs = spark.sql("select ST_Buffer_UDF(geos, 1.2) from envelope").collect()
    assert(rs[0][0] == 'POLYGON ((76.0974078351949 28.9406189138974,76.0941450901062 28.9703454241755,76.0741450901061 29.2103454241755,76.070582125475 29.2726267171628,76.0702606327722 29.3350090128529,76.0731814808438 29.397323720745,76.079336775988 29.4594024329966,76.0887098832881 29.5210773795525,76.101275471569 29.5821818815494,76.1169995818557 29.6425508017719,76.1358397191486 29.7020209909419,76.1577449672678 29.760431728636,76.1826561264563 29.8176251576381,76.2105058733692 29.8734467105547,76.2412189430177 29.927745527539,76.2747123321757 29.9803748639954,76.3108955236989 30.0311924871623,76.3496707311511 30.0800610605011,76.3909331630753 30.1268485148541,76.4345713061971 30.1714284053656,76.4804672267928 30.2136802532045,76.5284968894105 30.2534898711625,76.5785304920802 30.2907496722493,76.6304328171082 30.3253589604508,76.6840635965086 30.3572242028637,76.7392778910826 30.3862592824716,76.7959264821218 30.4123857308794,76.8538562746778 30.4355329403776,76.9129107113069 30.4556383547615,76.9729301951725 30.4726476383925,77.0337525213614 30.4865148230417,77.0952133152491 30.4972024321209,77.1571464767278 30.5046815819644,77.2193846290988 30.5089320598885,77.2817595714138 30.5099423788166,77.3441027330446 30.5077098083239,77.4062456292516 30.5022403820161,77.46802031652 30.4935488812235,77.5292598464334 30.481658795054,77.5897987168584 30.4666022569128,77.6494733192202 30.4484199576606,77.7994733192202 30.3984199576606,77.859060106788 30.3767928288753,77.9174216977522 30.3520492912892,77.9743952341763 30.3242583917088,78.0298217314878 30.2934976805998,78.0835465221248 30.2598529956829,78.1354196871355 30.2234182224034,78.1852964745266 30.1842950319444,78.2330377031939 30.1425925975132,78.2785101513067 30.0984272896938,78.3215869280632 30.0519223517149,78.362147827779 30.0032075555397,78.4000796653213 29.9524188397377,78.4352765919513 29.8996979301481,78.4676403906953 29.8451919443941,78.4970807504189 29.7890529813516,78.5235155178403 29.7314376967169,78.5468709267784 29.67250686586,78.5670818039979 29.6124249351806,78.5840917')

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    register_funcs(spark)

    run_st_point(spark)
    run_st_intersection(spark)
    run_st_isvalid(spark)
    run_st_equals(spark)
    run_st_touches(spark)
    run_st_overlaps(spark)
    run_st_crosses(spark)
    run_st_issimple(spark)
    run_st_geometry_type(spark)
    run_st_make_valid(spark)
    run_st_simplify_preserve_topology(spark)
    run_st_polygon_from_envelope(spark)
    run_st_contains(spark)
    run_st_intersects(spark)
    run_st_within(spark)
    run_st_distance(spark)
    run_st_area(spark)
    run_st_centroid(spark)
    run_st_length(spark)
    run_st_convexhull(spark)
    run_st_npoints(spark)
    run_st_envelope(spark)
    run_st_buffer(spark)

    spark.stop()
