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

from arctern.util import save_png
from arctern.util.vega import vega_pointmap, vega_heatmap, vega_choroplethmap

from arctern_pyspark import register_funcs
from arctern_pyspark import heatmap
from arctern_pyspark import pointmap
from arctern_pyspark import choroplethmap

from pyspark.sql import SparkSession

def draw_point_map(spark):
    # file 0_5M_nyc_taxi_and_building.csv could be obtained from arctern-turoial warehouse under zilliztech account. The link on github is https://github.com/zilliztech/arctern-tutorial
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    res = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), 'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')")

    vega = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.5, "EPSG:4326")
    res = pointmap(res, vega)
    save_png(res, '/tmp/pointmap.png')

    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")

def draw_heat_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    res = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')")

    res.show()
    vega = vega_heatmap(1024, 896, 10.0, [-73.998427, 40.730309, -73.954348, 40.780816], 'EPSG:4326')
    res = heatmap(res, vega)
    save_png(res, '/tmp/heatmap.png')

    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")

def draw_choropleth_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.createOrReplaceTempView("nyc_taxi")

    res = spark.sql("select buildingtext_dropoff as wkt, passenger_count as w from nyc_taxi")

    vega = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "blue_to_red", [2.5, 5], 1.0, 'EPSG:4326')
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
    draw_point_map(spark_session)
    draw_choropleth_map(spark_session)

    spark_session.stop()
