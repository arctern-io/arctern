from pyspark.sql import SparkSession
from zilliz_pyspark import register_funcs

def run_curve_z(spark):
    curve_z_df = spark.read.json("/tmp/z_curve.json").cache()
    curve_z_df.createOrReplaceTempView("curve_z")
    register_funcs(spark)
    hex_data = spark.sql("select my_plot(x, y) from curve_z").collect()[0][0]
    str_hex_data = str(hex_data)
    import binascii
    binary_string = binascii.unhexlify(str(hex_data))
    with open('/tmp/hex_curve_z.png', 'wb') as png:
        png.write(binary_string)


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Python TestPointmap") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_curve_z(spark)

    spark.stop()
