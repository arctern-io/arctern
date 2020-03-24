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

import sys
import cv2

from arctern.util import save_png
from arctern.util.vega import vega_pointmap, vega_heatmap, vega_choroplethmap

from arctern_pyspark import register_funcs
from arctern_pyspark import heatmap
from arctern_pyspark import pointmap
from arctern_pyspark import choroplethmap

from pyspark.sql import SparkSession

file_path = sys.path[0] + "/data/0_10000_nyc_taxi_and_building.csv"
baseline_png_path = sys.path[0] + "/draw_map/baseline_map/"
result_png_path = sys.path[0] + "/draw_map/result_map/"


def diff_png(baseline_png, compared_png, precision=0.0005):
    baseline_info = cv2.imread(baseline_png, cv2.IMREAD_UNCHANGED)
    compared_info = cv2.imread(compared_png, cv2.IMREAD_UNCHANGED)
    baseline_y, baseline_x = baseline_info.shape[0], baseline_info.shape[1]
    baseline_size = baseline_info.size

    compared_y, compared_x = compared_info.shape[0], compared_info.shape[1]
    compared_size = compared_info.size
    if compared_y != baseline_y or compared_x != baseline_x or compared_size != baseline_size:
        return False

    diff_point_num = 0
    for i in range(baseline_y):
        for j in range(baseline_x):
            baseline_rgba = baseline_info[i][j]
            compared_rgba = compared_info[i][j]

            baseline_rgba_len = len(baseline_rgba)
            compared_rgba_len = len(compared_rgba)
            if baseline_rgba_len != compared_rgba_len or baseline_rgba_len != 4:
                return False
            if compared_rgba[3] == baseline_rgba[3] and baseline_rgba[3] == 0:
                continue

            is_point_equal = True
            for k in range(3):
                tmp_diff = abs((int)(compared_rgba[k]) - (int)(baseline_rgba[k]))
                if tmp_diff > 1:
                    is_point_equal = False

            if not is_point_equal:
                diff_point_num += 1

    return ((float)(diff_point_num) / (float)(baseline_size)) <= precision


def draw_point_map(spark):
    # file 0_5M_nyc_taxi_and_building.csv could be obtained from arctern-turoial warehouse under zilliztech account. The link on github is https://github.com/zilliztech/arctern-tutorial
    # file 0_10000_nyc_taxi_and_building.csv is from file 0_5M_nyc_taxi_and_building.csv first 10000 lines
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, "
        "trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, "
        "dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, "
        "buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        file_path).cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    res = spark.sql(
        "select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), 'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')")

    # 1 size:1024*896, point_size: 3, opacity: 0.5, color: #2DEF4A
    vega_1 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.5, "EPSG:4326")
    baseline1 = pointmap(res, vega_1)
    point_map1_1 = pointmap(res, vega_1)
    point_map1_2 = pointmap(res, vega_1)

    baseline_png1 = baseline_png_path + "point_map_nyc_1.png"
    save_png(baseline1, baseline_png1)
    save_png(point_map1_1, result_png_path + "point_map_nyc_1-1.png")
    save_png(point_map1_2, result_png_path + "point_map_nyc_1-2.png")

    # 2 size:1024*896, point_size: 5, opacity: 0.5, color: #F50404
    vega_2 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 5, "F50404", 0.5, "EPSG:4326")
    baseline2 = pointmap(res, vega_2)
    point_map2_1 = pointmap(res, vega_2)
    point_map2_2 = pointmap(res, vega_2)

    baseline_png2 = baseline_png_path + "point_map_nyc_2.png"
    save_png(baseline2, baseline_png2)
    save_png(point_map2_1, result_png_path + "point_map_nyc_2-1.png")
    save_png(point_map2_2, result_png_path + "point_map_nyc_2-2.png")

    # 3 size:1024*896, point_size: 3, opacity: 1, color: #2DEF4A
    vega_3 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 1.0, "EPSG:4326")
    baseline3 = pointmap(res, vega_3)
    point_map3_1 = pointmap(res, vega_3)
    point_map3_2 = pointmap(res, vega_3)

    baseline_png3 = baseline_png_path + "point_map_nyc_3.png"
    save_png(baseline3, baseline_png3)
    save_png(point_map3_1, result_png_path + "point_map_nyc_3-1.png")
    save_png(point_map3_2, result_png_path + "point_map_nyc_3-2.png")

    # 4 size:1024*896, point_size: 3, opacity: 0, color: #2DEF4A
    vega_4 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.0, "EPSG:4326")
    baseline4 = pointmap(res, vega_4)
    point_map4_1 = pointmap(res, vega_4)
    point_map4_2 = pointmap(res, vega_4)

    baseline_png4 = baseline_png_path + "point_map_nyc_4.png"
    save_png(baseline4, baseline_png4)
    save_png(point_map4_1, result_png_path + "point_map_nyc_4-1.png")
    save_png(point_map4_2, result_png_path + "point_map_nyc_4-2.png")

    # 5 size:200*200, point_size: 3, opacity: 0.5, color: #2DEF4A
    vega_5 = vega_pointmap(200, 200, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.5, "EPSG:4326")
    baseline5 = pointmap(res, vega_5)
    point_map5_1 = pointmap(res, vega_5)
    point_map5_2 = pointmap(res, vega_5)

    baseline_png5 = baseline_png_path + "point_map_nyc_5.png"
    save_png(baseline5, baseline_png5)
    save_png(point_map5_1, result_png_path + "point_map_nyc_5-1.png")
    save_png(point_map5_2, result_png_path + "point_map_nyc_5-2.png")

    spark.catalog.dropGlobalTempView("nyc_taxi")

    assert diff_png(baseline_png1, result_png_path + "point_map_nyc_1-1.png", 0.00005)
    assert diff_png(baseline_png1, result_png_path + "point_map_nyc_1-2.png", 0.00005)
    assert diff_png(baseline_png2, result_png_path + "point_map_nyc_2-1.png", 0.00005)
    assert diff_png(baseline_png2, result_png_path + "point_map_nyc_2-2.png", 0.00005)
    assert diff_png(baseline_png3, result_png_path + "point_map_nyc_3-1.png", 0.00005)
    assert diff_png(baseline_png3, result_png_path + "point_map_nyc_3-2.png", 0.00005)
    assert diff_png(baseline_png4, result_png_path + "point_map_nyc_4-1.png", 0.00005)
    assert diff_png(baseline_png4, result_png_path + "point_map_nyc_4-2.png", 0.00005)
    assert diff_png(baseline_png4, result_png_path + "point_map_nyc_4-1.png", 0.00005)
    assert diff_png(baseline_png4, result_png_path + "point_map_nyc_4-2.png", 0.00005)


def draw_heat_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, "
        "trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, "
        "dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, "
        "buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        file_path).cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    res = spark.sql(
        "select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')")

    # 1 size:1024*896, map_scale: 10.0
    vega_1 = vega_heatmap(200, 200, 10.0, [-73.998427, 40.730309, -73.954348, 40.780816], 'EPSG:4326')
    baseline1 = heatmap(res, vega_1)
    heat_map1_1 = heatmap(res, vega_1)
    heat_map1_2 = heatmap(res, vega_1)

    baseline_png1 = baseline_png_path + "heat_map_nyc_1.png"
    save_png(baseline1, baseline_png1)
    save_png(heat_map1_1, result_png_path + "heat_map_nyc_1-1.png")
    save_png(heat_map1_2, result_png_path + "heat_map_nyc_1-2.png")

    # 2 size:1024*896, map_scale: 0.0
    vega_2 = vega_heatmap(1024, 896, 0.0, [-73.998427, 40.730309, -73.954348, 40.780816], 'EPSG:4326')
    baseline2 = heatmap(res, vega_2)
    heat_map2_1 = heatmap(res, vega_2)
    heat_map2_2 = heatmap(res, vega_2)

    baseline_png2 = baseline_png_path + "heat_map_nyc_2.png"
    save_png(baseline2, baseline_png2)
    save_png(heat_map2_1, result_png_path + "heat_map_nyc_2-1.png")
    save_png(heat_map2_2, result_png_path + "heat_map_nyc_2-2.png")

    # 3 size:1024*896, map_scale: 12.0
    vega_3 = vega_heatmap(1024, 896, 12.0, [-73.998427, 40.730309, -73.954348, 40.780816], 'EPSG:4326')
    baseline3 = heatmap(res, vega_3)
    heat_map3_1 = heatmap(res, vega_3)
    heat_map3_2 = heatmap(res, vega_3)

    baseline_png3 = baseline_png_path + "heat_map_nyc_3.png"
    save_png(baseline3, baseline_png3)
    save_png(heat_map3_1, result_png_path + "heat_map_nyc_3-1.png")
    save_png(heat_map3_2, result_png_path + "heat_map_nyc_3-2.png")

    assert diff_png(baseline_png1, result_png_path + "heat_map_nyc_1-1.png", 0.1)
    assert diff_png(baseline_png1, result_png_path + "heat_map_nyc_1-2.png", 0.1)
    assert diff_png(baseline_png2, result_png_path + "heat_map_nyc_2-1.png", 0.1)
    assert diff_png(baseline_png2, result_png_path + "heat_map_nyc_2-2.png", 0.1)
    assert diff_png(baseline_png3, result_png_path + "heat_map_nyc_3-1.png", 0.2)
    assert diff_png(baseline_png3, result_png_path + "heat_map_nyc_3-2.png", 0.2)

    spark.catalog.dropGlobalTempView("nyc_taxi")


def draw_choropleth_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, "
        "trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, "
        "dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, "
        "buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        file_path).cache()
    df.createOrReplaceTempView("nyc_taxi")

    res = spark.sql("select buildingtext_dropoff as wkt, passenger_count as w from nyc_taxi")

    # 1 blue_to_red,
    vega_1 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "blue_to_red", [2.5, 5],
                                1.0, 'EPSG:4326')
    baseline1 = choroplethmap(res, vega_1)
    choropleth_map1_1 = choroplethmap(res, vega_1)
    choropleth_map1_2 = choroplethmap(res, vega_1)

    baseline_png1 = baseline_png_path + "choropleth_map_nyc_1.png"
    save_png(baseline1, baseline_png1)
    save_png(choropleth_map1_1, result_png_path + "choropleth_map_nyc_1-1.png")
    save_png(choropleth_map1_2, result_png_path + "choropleth_map_nyc_1-2.png")

    # 2
    vega_2 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "blue_to_red", [2.5, 5],
                                0.0, 'EPSG:4326')
    baseline2 = choroplethmap(res, vega_2)
    choropleth_map2_1 = choroplethmap(res, vega_2)
    choropleth_map2_2 = choroplethmap(res, vega_2)

    baseline_png2 = baseline_png_path + "choropleth_map_nyc_2.png"
    save_png(baseline2, baseline_png2)
    save_png(choropleth_map2_1, result_png_path + "choropleth_map_nyc_2-1.png")
    save_png(choropleth_map2_2, result_png_path + "choropleth_map_nyc_2-2.png")

    # 3
    vega_3 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "blue_to_red", [1, 10000],
                                1.0, 'EPSG:4326')
    baseline3 = choroplethmap(res, vega_3)
    choropleth_map3_1 = choroplethmap(res, vega_3)
    choropleth_map3_2 = choroplethmap(res, vega_3)

    baseline_png3 = baseline_png_path + "choropleth_map_nyc_3.png"
    save_png(baseline3, baseline_png3)
    save_png(choropleth_map3_1, result_png_path + "choropleth_map_nyc_3-1.png")
    save_png(choropleth_map3_2, result_png_path + "choropleth_map_nyc_3-2.png")

    # 4
    vega_4 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "skyblue_to_white",
                                [2.5, 5],
                                1.0, 'EPSG:4326')
    baseline4 = choroplethmap(res, vega_4)
    choropleth_map4_1 = choroplethmap(res, vega_4)
    choropleth_map4_2 = choroplethmap(res, vega_4)

    baseline_png4 = baseline_png_path + "choropleth_map_nyc_4.png"
    save_png(baseline4, baseline_png4)
    save_png(choropleth_map4_1, result_png_path + "choropleth_map_nyc_4-1.png")
    save_png(choropleth_map4_2, result_png_path + "choropleth_map_nyc_4-2.png")

    # 5
    vega_5 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "purple_to_yellow",
                                [2.5, 5],
                                1.0, 'EPSG:4326')
    baseline5 = choroplethmap(res, vega_5)
    choropleth_map5_1 = choroplethmap(res, vega_5)
    choropleth_map5_2 = choroplethmap(res, vega_5)

    baseline_png5 = baseline_png_path + "choropleth_map_nyc_5.png"
    save_png(baseline5, baseline_png5)
    save_png(choropleth_map5_1, result_png_path + "choropleth_map_nyc_5-1.png")
    save_png(choropleth_map5_2, result_png_path + "choropleth_map_nyc_5-2.png")

    # 6
    vega_6 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "red_transparency",
                                [2.5, 5],
                                1.0, 'EPSG:4326')
    baseline6 = choroplethmap(res, vega_6)
    choropleth_map6_1 = choroplethmap(res, vega_6)
    choropleth_map6_2 = choroplethmap(res, vega_6)

    baseline_png6 = baseline_png_path + "choropleth_map_nyc_6.png"
    save_png(baseline6, baseline_png6)
    save_png(choropleth_map6_1, result_png_path + "choropleth_map_nyc_6-1.png")
    save_png(choropleth_map6_2, result_png_path + "choropleth_map_nyc_6-2.png")

    # 7
    vega_7 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "blue_transparency",
                                [2.5, 5],
                                1.0, 'EPSG:4326')
    baseline7 = choroplethmap(res, vega_7)
    choropleth_map7_1 = choroplethmap(res, vega_7)
    choropleth_map7_2 = choroplethmap(res, vega_7)

    baseline_png7 = baseline_png_path + "choropleth_map_nyc_7.png"
    save_png(baseline7, baseline_png7)
    save_png(choropleth_map7_1, result_png_path + "choropleth_map_nyc_7-1.png")
    save_png(choropleth_map7_2, result_png_path + "choropleth_map_nyc_7-2.png")

    # 8
    vega_8 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "blue_green_yellow",
                                [2.5, 5],
                                1.0, 'EPSG:4326')
    baseline8 = choroplethmap(res, vega_8)
    choropleth_map8_1 = choroplethmap(res, vega_8)
    choropleth_map8_2 = choroplethmap(res, vega_8)

    baseline_png8 = baseline_png_path + "choropleth_map_nyc_8.png"
    save_png(baseline8, baseline_png8)
    save_png(choropleth_map8_1, result_png_path + "choropleth_map_nyc_8-1.png")
    save_png(choropleth_map8_2, result_png_path + "choropleth_map_nyc_8-2.png")

    # 9
    vega_9 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "white_blue",
                                [2.5, 5],
                                1.0, 'EPSG:4326')
    baseline9 = choroplethmap(res, vega_9)
    choropleth_map9_1 = choroplethmap(res, vega_9)
    choropleth_map9_2 = choroplethmap(res, vega_9)

    baseline_png9 = baseline_png_path + "choropleth_map_nyc_9.png"
    save_png(baseline9, baseline_png9)
    save_png(choropleth_map9_1, result_png_path + "choropleth_map_nyc_9-1.png")
    save_png(choropleth_map9_2, result_png_path + "choropleth_map_nyc_9-2.png")

    # 10
    vega_10 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "blue_white_red",
                                 [2.5, 5],
                                 1.0, 'EPSG:4326')
    baseline10 = choroplethmap(res, vega_10)
    choropleth_map10_1 = choroplethmap(res, vega_10)
    choropleth_map10_2 = choroplethmap(res, vega_10)

    baseline_png10 = baseline_png_path + "choropleth_map_nyc_10.png"
    save_png(baseline10, baseline_png10)
    save_png(choropleth_map10_1, result_png_path + "choropleth_map_nyc_10-1.png")
    save_png(choropleth_map10_2, result_png_path + "choropleth_map_nyc_10-2.png")

    # 11
    vega_11 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "green_yellow_red",
                                 [2.5, 5],
                                 1.0, 'EPSG:4326')
    baseline11 = choroplethmap(res, vega_11)
    choropleth_map11_1 = choroplethmap(res, vega_11)
    choropleth_map11_2 = choroplethmap(res, vega_11)

    baseline_png11 = baseline_png_path + "choropleth_map_nyc_11.png"
    save_png(baseline11, baseline_png11)
    save_png(choropleth_map11_1, result_png_path + "choropleth_map_nyc_11-1.png")
    save_png(choropleth_map11_2, result_png_path + "choropleth_map_nyc_11-2.png")

    # 12
    vega_12 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "purple_to_yellow",
                                 [2.5, 5],
                                 1.0, 'EPSG:4326')
    baseline12 = choroplethmap(res, vega_12)
    choropleth_map12_1 = choroplethmap(res, vega_12)
    choropleth_map12_2 = choroplethmap(res, vega_12)

    baseline_png12 = result_png_path + "choropleth_map_nyc_12.png"
    save_png(baseline12, baseline_png12)
    save_png(choropleth_map12_1, result_png_path + "choropleth_map_nyc_12-1.png")
    save_png(choropleth_map12_2, result_png_path + "choropleth_map_nyc_12-2.png")

    # 13
    vega_13 = vega_choroplethmap(256, 256, [-73.994092, 40.753893, -73.977588, 40.759642], "purple_to_yellow",
                                 [2.5, 5],
                                 1.0, 'EPSG:4326')
    baseline13 = choroplethmap(res, vega_13)
    choropleth_map13_1 = choroplethmap(res, vega_13)
    choropleth_map13_2 = choroplethmap(res, vega_13)

    baseline_png13 = baseline_png_path + "choropleth_map_nyc_13.png"
    save_png(baseline13, baseline_png13)
    save_png(choropleth_map13_1, result_png_path + "choropleth_map_nyc_13-1.png")
    save_png(choropleth_map13_2, result_png_path + "choropleth_map_nyc_13-2.png")

    # 14
    vega_14 = vega_choroplethmap(200, 200, [-73.994092, 40.753893, -73.977588, 40.759642], "purple_to_yellow",
                                 [2.5, 5],
                                 1.0, 'EPSG:4326')
    baseline14 = choroplethmap(res, vega_14)
    choropleth_map14_1 = choroplethmap(res, vega_14)
    choropleth_map14_2 = choroplethmap(res, vega_14)

    baseline_png14 = baseline_png_path + "choropleth_map_nyc_14.png"
    save_png(baseline14, baseline_png14)
    save_png(choropleth_map14_1, result_png_path + "choropleth_map_nyc_14-1.png")
    save_png(choropleth_map14_2, result_png_path + "choropleth_map_nyc_14-2.png")

    spark.catalog.dropGlobalTempView("nyc_taxi")

    assert diff_png(baseline_png1, result_png_path + "choropleth_map_nyc_1-1.png", 0.00005)
    assert diff_png(baseline_png1, result_png_path + "choropleth_map_nyc_1-2.png", 0.00005)
    assert diff_png(baseline_png2, result_png_path + "choropleth_map_nyc_2-1.png", 0.00005)
    assert diff_png(baseline_png2, result_png_path + "choropleth_map_nyc_2-2.png", 0.00005)
    assert diff_png(baseline_png3, result_png_path + "choropleth_map_nyc_3-1.png", 0.00005)
    assert diff_png(baseline_png3, result_png_path + "choropleth_map_nyc_3-2.png", 0.00005)
    assert diff_png(baseline_png4, result_png_path + "choropleth_map_nyc_4-1.png", 0.00005)
    assert diff_png(baseline_png4, result_png_path + "choropleth_map_nyc_4-2.png", 0.00005)
    assert diff_png(baseline_png5, result_png_path + "choropleth_map_nyc_5-1.png", 0.00005)
    assert diff_png(baseline_png5, result_png_path + "choropleth_map_nyc_5-2.png", 0.00005)
    assert diff_png(baseline_png6, result_png_path + "choropleth_map_nyc_6-1.png", 0.00005)
    assert diff_png(baseline_png6, result_png_path + "choropleth_map_nyc_6-2.png", 0.00005)
    assert diff_png(baseline_png7, result_png_path + "choropleth_map_nyc_7-1.png", 0.00005)
    assert diff_png(baseline_png7, result_png_path + "choropleth_map_nyc_7-2.png", 0.00005)
    assert diff_png(baseline_png8, result_png_path + "choropleth_map_nyc_8-1.png", 0.00005)
    assert diff_png(baseline_png8, result_png_path + "choropleth_map_nyc_8-2.png", 0.00005)
    assert diff_png(baseline_png9, result_png_path + "choropleth_map_nyc_9-1.png", 0.00005)
    assert diff_png(baseline_png9, result_png_path + "choropleth_map_nyc_9-2.png", 0.00005)
    assert diff_png(baseline_png10, result_png_path + "choropleth_map_nyc_10-1.png", 0.00005)
    assert diff_png(baseline_png10, result_png_path + "choropleth_map_nyc_10-2.png", 0.00005)
    assert diff_png(baseline_png11, result_png_path + "choropleth_map_nyc_11-1.png", 0.00005)
    assert diff_png(baseline_png11, result_png_path + "choropleth_map_nyc_11-2.png", 0.00005)
    assert diff_png(baseline_png12, result_png_path + "choropleth_map_nyc_12-1.png", 0.00005)
    assert diff_png(baseline_png12, result_png_path + "choropleth_map_nyc_12-2.png", 0.00005)
    assert diff_png(baseline_png13, result_png_path + "choropleth_map_nyc_13-1.png", 0.00005)
    assert diff_png(baseline_png13, result_png_path + "choropleth_map_nyc_13-2.png", 0.00005)
    assert diff_png(baseline_png14, result_png_path + "choropleth_map_nyc_14-1.png", 0.00005)
    assert diff_png(baseline_png14, result_png_path + "choropleth_map_nyc_14-2.png", 0.00005)


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
