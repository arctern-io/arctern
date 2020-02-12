from pyspark.sql import SparkSession
from zilliz_pyspark import register_funcs


def run_st_intersection(spark):
    test_df = spark.read.json("/tmp/intersection.json").cache()
    test_df.createOrReplaceTempView("intersection")
    register_funcs(spark)
    spark.sql("select ST_Intersection_UDF(left, right) from intersection").show()

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_st_intersection(spark)

    spark.stop()
