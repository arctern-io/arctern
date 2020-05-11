# pylint: disable=wrong-import-order

from pyspark.sql.types import StructType, StructField, LongType, StringType
from pyspark.sql import SparkSession

import matplotlib.pyplot as plt

from arctern_pyspark import register_funcs
from arctern_pyspark import plot

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("plot_test") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    register_funcs(spark)

    raw_data = []
    raw_data.extend([(0, 'polygon((0 0,0 1,1 1,1 0,0 0))')])
    raw_data.extend([(1, 'linestring(0 0,0 1,1 1,1 0,0 0)')])
    raw_data.extend([(2, 'point(2 2)')])

    wkt_collect = "GEOMETRYCOLLECTION(" \
                  "MULTIPOLYGON (((0 0,0 1,1 1,1 0,0 0)),((1 1,1 2,2 2,2 1,1 1)))," \
                  "POLYGON((3 3,3 4,4 4,4 3,3 3))," \
                  "LINESTRING(0 8,5 5,8 0)," \
                  "POINT(4 7)," \
                  "MULTILINESTRING ((1 1,1 2),(2 4,1 9,1 8))," \
                  "MULTIPOINT (6 8,5 7)" \
                  ")"
    raw_data.extend([(3, wkt_collect)])


    raw_schema = StructType([
                            StructField('idx', LongType(), False),
                            StructField('geo', StringType(), False)
                            ])

    df = spark.createDataFrame(data=raw_data, schema=raw_schema)
    df.createOrReplaceTempView("geoms")
    df2 = spark.sql("select st_geomfromtext(geo) from geoms")

    fig, ax = plt.subplots()
    plot(ax, df2)
    ax.grid()
    fig.savefig("/tmp/plot_test.png")
    spark.stop()
