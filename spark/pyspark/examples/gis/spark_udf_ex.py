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

def run_st_equals(spark):
    test_data = []
    test_data.extend([('LINESTRING(0 0, 10 10)', 'LINESTRING(0 0, 5 5, 10 10)')])
    test_data.extend([('LINESTRING(10 10, 0 0)', 'LINESTRING(0 0, 5 5, 10 10)')])
    equals_df = spark.createDataFrame(data = test_data, schema = ["left", "right"]).cache()
    equals_df.createOrReplaceTempView("equals")
    rs = spark.sql("select ST_Equals_UDF(left, right) from equals").collect()
    assert(not rs[0][0])
    assert(not rs[1][0])

# TODO: ST_Touches due to runtime error

def run_st_isvalid(spark):
    test_data = []
    test_data.extend([('POINT (30 10)')])
    test_data.extend([('i am not a valid geometry')])
    valid_df = spark.createDataFrame(data = test_data, schema = ['geos']).cache()
    valid_df.createOrReplaceTempView("valid")
    rs = spark.sql("select ST_IsValid_UDF(geos) from valid").collect()
    assert(rs[0][0])
    assert(not rs[1][0])

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

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    register_funcs(spark)

    run_st_point(spark)

    spark.stop()
