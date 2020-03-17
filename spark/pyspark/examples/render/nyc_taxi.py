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

from arctern.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d
from arctern.util.vega.heat_map.vega_heat_map import VegaHeatMap
from arctern.util.vega.choropleth_map.choropleth_map import VegaChoroplethMap

from arctern_pyspark import register_funcs
from arctern_pyspark import heatmap
from arctern_pyspark import pointmap
from arctern_pyspark import choroplethmap
from arctern_pyspark import save_png

from pyspark.sql import SparkSession
from pyspark.sql.types import *

def draw_point_map(spark):
    # file 0_5M_nyc_taxi_and_building.csv could be obtained from arctern-turoial warehouse under zilliztech account. The link on github is https://github.com/zilliztech/arctern-tutorial
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.show(20, False)
    df.createOrReplaceTempView("nyc_taxi")
    # df.createOrReplaceGlobalTempView("nyc_taxi")

    res = spark.sql("select pickup_latitude as x, pickup_longitude as y  from nyc_taxi")
    res.printSchema()
    res.createOrReplaceTempView("pickup")

    register_funcs(spark)
    res = spark.sql(
        "select ST_Transform(ST_Point(x, y), 'EPSG:4326','EPSG:3857' ) as pickup_point from pickup")
    res.show(20, False)
    res.createOrReplaceTempView("project")

    res = spark.sql(
        "select Projection(pickup_point, 'POINT (4534000 -12510000)', 'POINT (4538000 -12513000)', 1024, 896) as point from project")
    res.show(20, False)

    vega_point_map = VegaCircle2d(1900, 1410, 3, "#2DEF4A", 0.5)
    vega = vega_point_map.build()
    res = pointmap(res, vega)
    save_png(res, '/tmp/pointmap.png')

    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")

def draw_heat_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_build.csv").cache()
    # df.show(20, False)
    df.createOrReplaceTempView("nyc_taxi")
    # df.createOrReplaceGlobalTempView("nyc_taxi")

    res = spark.sql("select pickup_latitude as x, pickup_longitude as y, passenger_count as w from nyc_taxi")
    res.printSchema()
    res.createOrReplaceTempView("pickup")

    register_funcs(spark)
    res = spark.sql(
        "select ST_Transform(ST_Point(x, y), 'EPSG:4326','EPSG:3857' ) as pickup_point, w from pickup")
    # res.show(20, False)
    res.createOrReplaceTempView("project")

    res = spark.sql(
        "select Projection(pickup_point, 'POINT (4534000 -12510000)', 'POINT (4538000 -12513000)', 1024, 896) as point, w from project")
    # res.show(20, False)

    vega_heat_map = VegaHeatMap(300, 200, 10.0)
    vega = vega_heat_map.build()
    res = heatmap(res, vega)
    save_png(res, '/tmp/heatmap.png')

    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")

def draw_choropleth_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.show(20, False)
    df.createOrReplaceTempView("nyc_taxi")
    # df.createOrReplaceGlobalTempView("nyc_taxi")

    res = spark.sql("select buildingtext_dropoff as wkt, passenger_count as w from nyc_taxi")
    res.printSchema()
    res.createOrReplaceTempView("pickup")
        
    vega_choropleth_map = VegaChoroplethMap(1900, 1410, [-73.984092, 40.753893, -73.977588, 40.756342], "blue_to_red", [2.5, 5], 1.0)
    vega = vega_choropleth_map.build()
    res = choroplethmap(res, vega)
    save_png(res, '/tmp/choroplethmap.png')

    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")


if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python Testmap") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    draw_heat_map(spark_session)
#    draw_point_map(spark_session)
#    draw_choropleth_map(spark_session)
    spark_session.stop()
