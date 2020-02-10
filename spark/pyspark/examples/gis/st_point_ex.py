from pyspark.sql import SparkSession
from zilliz_pyspark import register_funcs


def run_st_point(spark):
    points_df = spark.read.json("/tmp/points.json").cache()
    points_df.createOrReplaceTempView("points")
    register_funcs(spark)
    spark.sql("select ST_Point_UDF(x, y) from points").show()

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_st_point(spark)

    spark.stop()
