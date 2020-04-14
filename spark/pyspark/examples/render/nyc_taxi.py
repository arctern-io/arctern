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
from arctern.util.vega import vega_pointmap, vega_heatmap, vega_choroplethmap, vega_weighted_pointmap

from arctern_pyspark import register_funcs
from arctern_pyspark import heatmap
from arctern_pyspark import pointmap
from arctern_pyspark import choroplethmap
from arctern_pyspark import weighted_pointmap

from pyspark.sql import SparkSession

import time

def draw_point_map(spark):
    start_time = time.time()
    # file 0_5M_nyc_taxi_and_building.csv could be obtained from arctern-turoial warehouse under zilliztech account. The link on github is https://github.com/zilliztech/arctern-tutorial
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    res = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    vega = vega_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], point_size=3, point_color="#2DEF4A", opacity=0.5, coordinate_system="EPSG:4326")
    res = pointmap(vega, res)
    save_png(res, '/tmp/pointmap.png')

    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")
    print("--- %s seconds ---" % (time.time() - start_time))

def draw_weighted_point_map(spark):
    start_time = time.time()
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)

    # single color and single stroke width
    res1 = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")
    vega1 = vega_weighted_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#87CEEB"], opacity=1.0, coordinate_system="EPSG:4326")
    res1 = weighted_pointmap(vega1, res1)
    save_png(res1, '/tmp/weighted_pointmap_0_0.png')

    # multiple color and single stroke width
    res2 = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")
    vega2 = vega_weighted_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF", "#FF0000"], color_bound=[0, 2], opacity=1.0, coordinate_system="EPSG:4326")
    res2 = weighted_pointmap(vega2, res2)
    save_png(res2, '/tmp/weighted_pointmap_1_0.png')

    # single color and multiple stroke width
    res3 = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point, fare_amount as s from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")
    vega3 = vega_weighted_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#87CEEB"], size_bound=[0, 10], opacity=1.0, coordinate_system="EPSG:4326")
    res3 = weighted_pointmap(vega3, res3)
    save_png(res3, '/tmp/weighted_pointmap_0_1.png')

    # multiple color and multiple stroke width
    res4 = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c, fare_amount as s from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")
    vega4 = vega_weighted_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF", "#FF0000"], color_bound=[0, 2], size_bound=[0, 10], opacity=1.0, coordinate_system="EPSG:4326")
    res4 = weighted_pointmap(vega4, res4)
    save_png(res4, '/tmp/weighted_pointmap_1_1.png')



    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")
    print("--- %s seconds ---" % (time.time() - start_time))

def draw_heat_map(spark):
    start_time = time.time()
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    res = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    res.show()

    vega = vega_heatmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], map_zoom_level=10.0, coordinate_system='EPSG:4326')
    res = heatmap(vega, res)
    save_png(res, '/tmp/heatmap.png')

    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")
    print("--- %s seconds ---" % (time.time() - start_time))

def draw_choropleth_map(spark):
    start_time = time.time()
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        "file:///tmp/0_5M_nyc_taxi_and_building.csv").cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    res = spark.sql("select ST_GeomFromText(buildingtext_dropoff) as polygon, passenger_count as w from nyc_taxi")

    vega1 = vega_choroplethmap(1900, 1410, bounding_box=[-73.994092, 40.753893, -73.977588, 40.759642], color_gradient=["#0000FF", "#FF0000"], color_bound=[2.5, 5], opacity=1.0, coordinate_system='EPSG:4326') 
    res1 = choroplethmap(vega1, res)
    save_png(res1, '/tmp/choroplethmap1.png')

    spark.sql("show tables").show()
    spark.catalog.dropGlobalTempView("nyc_taxi")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python Testmap") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    draw_heat_map(spark_session)
    draw_point_map(spark_session)
    draw_choropleth_map(spark_session)
    draw_weighted_point_map(spark_session)

    spark_session.stop()
