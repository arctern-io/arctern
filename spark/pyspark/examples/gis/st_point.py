from pyspark.sql import SparkSession
from zilliz_gis.register import register_funcs


def example(spark):
    df = spark.read.json("/opt/spark-3.0.0-preview/python/points.json")
    df.createOrReplaceTempView("points")
    spark.sql("select ST_Point(age, age) from points").show()


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    register_funcs(spark)
    example(spark)
    spark.stop()