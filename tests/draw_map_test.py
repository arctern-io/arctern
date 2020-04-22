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

# pylint: disable=c-extension-no-member
import cv2

from arctern.util import save_png
from arctern.util.vega import vega_pointmap, vega_heatmap, vega_choroplethmap, vega_weighted_pointmap, vega_icon

from pyspark.sql import SparkSession

from arctern_pyspark import register_funcs
from arctern_pyspark import heatmap
from arctern_pyspark import pointmap
from arctern_pyspark import choroplethmap
from arctern_pyspark import weighted_pointmap
from arctern_pyspark import icon_viz

file_path = sys.path[0] + "/data/0_10000_nyc_taxi_and_building.csv"
png_path = sys.path[0] + "/draw_map/"

def run_diff_png(baseline_png, compared_png, precision=0.0005):
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

# pylint: disable=too-many-statements
def run_test_point_map(spark):
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
        "select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    # 1 size:1024*896, point_size: 3, opacity: 0.5, color: #2DEF4A(green)
    vega_1 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.5, "EPSG:4326")
    baseline1 = pointmap(vega_1, res)
    point_map1_1 = pointmap(vega_1, res)
    point_map1_2 = pointmap(vega_1, res)

    baseline_png1 = png_path + "point_map_nyc_1.png"
    save_png(baseline1, baseline_png1)
    save_png(point_map1_1, png_path + "test_point_map_nyc_1-1.png")
    save_png(point_map1_2, png_path + "test_point_map_nyc_1-2.png")

    # 2 #F50404(red)
    vega_2 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 5, "#F50404", 0.5, "EPSG:4326")
    baseline2 = pointmap(vega_2, res)
    point_map2_1 = pointmap(vega_2, res)
    point_map2_2 = pointmap(vega_2, res)

    baseline_png2 = png_path + "point_map_nyc_2.png"
    save_png(baseline2, baseline_png2)
    save_png(point_map2_1, png_path + "test_point_map_nyc_2-1.png")
    save_png(point_map2_2, png_path + "test_point_map_nyc_2-2.png")

    # 3 color: #1455EE(blue)
    vega_3 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 5, "#1455EE", 0.5, "EPSG:4326")
    baseline3 = pointmap(vega_3, res)
    point_map3_1 = pointmap(vega_3, res)
    point_map3_2 = pointmap(vega_3, res)

    baseline_png3 = png_path + "point_map_nyc_3.png"
    save_png(baseline3, baseline_png3)
    save_png(point_map3_1, png_path + "test_point_map_nyc_3-1.png")
    save_png(point_map3_2, png_path + "test_point_map_nyc_3-2.png")

    # 4 size:1024*896, point_size: 3, opacity: 1, color: #2DEF4A
    vega_4 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 1.0, "EPSG:4326")
    baseline4 = pointmap(vega_4, res)
    point_map4_1 = pointmap(vega_4, res)
    point_map4_2 = pointmap(vega_4, res)

    baseline_png4 = png_path + "point_map_nyc_4.png"
    save_png(baseline4, baseline_png4)
    save_png(point_map4_1, png_path + "test_point_map_nyc_4-1.png")
    save_png(point_map4_2, png_path + "test_point_map_nyc_4-2.png")

    # 5 size:1024*896, point_size: 3, opacity: 0, color: #2DEF4A
    vega_5 = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.0, "EPSG:4326")
    baseline5 = pointmap(vega_5, res)
    point_map5_1 = pointmap(vega_5, res)
    point_map5_2 = pointmap(vega_5, res)

    baseline_png5 = png_path + "point_map_nyc_5.png"
    save_png(baseline5, baseline_png5)
    save_png(point_map5_1, png_path + "test_point_map_nyc_5-1.png")
    save_png(point_map5_2, png_path + "test_point_map_nyc_5-2.png")

    # 6 size:200*200, point_size: 3, opacity: 0.5, color: #2DEF4A
    vega_6 = vega_pointmap(200, 200, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.5, "EPSG:4326")
    baseline6 = pointmap(vega_6, res)
    point_map6_1 = pointmap(vega_6, res)
    point_map6_2 = pointmap(vega_6, res)

    baseline_png6 = png_path + "point_map_nyc_6.png"
    save_png(baseline6, baseline_png6)
    save_png(point_map6_1, png_path + "test_point_map_nyc_6-1.png")
    save_png(point_map6_2, png_path + "test_point_map_nyc_6-2.png")

    spark.catalog.dropGlobalTempView("nyc_taxi")

    assert run_diff_png(baseline_png1, png_path + "test_point_map_nyc_1-1.png")
    assert run_diff_png(baseline_png1, png_path + "test_point_map_nyc_1-2.png")
    assert run_diff_png(baseline_png2, png_path + "test_point_map_nyc_2-1.png")
    assert run_diff_png(baseline_png2, png_path + "test_point_map_nyc_2-2.png")
    assert run_diff_png(baseline_png3, png_path + "test_point_map_nyc_3-1.png")
    assert run_diff_png(baseline_png3, png_path + "test_point_map_nyc_3-2.png")
    assert run_diff_png(baseline_png4, png_path + "test_point_map_nyc_4-1.png")
    assert run_diff_png(baseline_png4, png_path + "test_point_map_nyc_4-2.png")
    assert run_diff_png(baseline_png5, png_path + "test_point_map_nyc_5-1.png")
    assert run_diff_png(baseline_png5, png_path + "test_point_map_nyc_5-2.png")
    assert run_diff_png(baseline_png6, png_path + "test_point_map_nyc_6-1.png")
    assert run_diff_png(baseline_png6, png_path + "test_point_map_nyc_6-2.png")

# pylint: disable=too-many-statements
def run_test_weighted_point_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, "
        "trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, "
        "dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, "
        "buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        file_path).cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    # 1 single color; single point size
    res1 = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    # 1.1 opacity = 1.0, color_ruler: [0, 2], color: #EE3814(red)
    vega1_1 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#EE3814"], [0, 2],
                                     [10], 1.0, "EPSG:4326")
    baseline1 = weighted_pointmap(vega1_1, res1)
    weighted_point_map1_1_1 = weighted_pointmap(vega1_1, res1)
    weighted_point_map1_1_2 = weighted_pointmap(vega1_1, res1)

    baseline_png1_1 = png_path + "weighted_point_map_nyc_1_1.png"
    save_png(baseline1, baseline_png1_1)
    save_png(weighted_point_map1_1_1, png_path + "test_weighted_point_map_nyc_1_1-1.png")
    save_png(weighted_point_map1_1_2, png_path + "test_weighted_point_map_nyc_1_1-2.png")

    # 1.2 opacity = 0.0, color_ruler: [0, 2], color: #EE3814(red)
    vega1_2 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#EE3814"], [0, 2],
                                     [10], 0.0, "EPSG:4326")
    baseline1_2 = weighted_pointmap(vega1_2, res1)
    weighted_point_map1_2_1 = weighted_pointmap(vega1_2, res1)
    weighted_point_map1_2_2 = weighted_pointmap(vega1_2, res1)

    baseline_png1_2 = png_path + "weighted_point_map_nyc_1_2.png"
    save_png(baseline1_2, baseline_png1_2)
    save_png(weighted_point_map1_2_1, png_path + "test_weighted_point_map_nyc_1_2-1.png")
    save_png(weighted_point_map1_2_2, png_path + "test_weighted_point_map_nyc_1_2-2.png")

    # 1.3 opacity = 1.0, color_ruler: [0, 100], color: #14EE47(green)
    vega1_3 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#14EE47"], [0, 100],
                                     [5], 1.0, "EPSG:4326")
    baseline1_3 = weighted_pointmap(vega1_3, res1)
    weighted_point_map1_3_1 = weighted_pointmap(vega1_3, res1)
    weighted_point_map1_3_2 = weighted_pointmap(vega1_3, res1)

    baseline_png1_3 = png_path + "weighted_point_map_nyc_1_3.png"
    save_png(baseline1_3, baseline_png1_3)
    save_png(weighted_point_map1_3_1, png_path + "test_weighted_point_map_nyc_1_3-1.png")
    save_png(weighted_point_map1_3_2, png_path + "test_weighted_point_map_nyc_1_3-2.png")

    # 1.4 opacity = 0.5, color_ruler: [0, 2], color: #1221EE, stroke_ruler: [5]
    vega1_4 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#1221EE"], [0, 2],
                                     [5], 0.5, "EPSG:4326")
    baseline1_4 = weighted_pointmap(vega1_4, res1)
    weighted_point_map1_4_1 = weighted_pointmap(vega1_4, res1)
    weighted_point_map1_4_2 = weighted_pointmap(vega1_4, res1)

    baseline_png1_4 = png_path + "weighted_point_map_nyc_1_4.png"
    save_png(baseline1_4, baseline_png1_4)
    save_png(weighted_point_map1_4_1, png_path + "test_weighted_point_map_nyc_1_4-1.png")
    save_png(weighted_point_map1_4_2, png_path + "test_weighted_point_map_nyc_1_4-2.png")

    # 1.5 size: 200*200, opacity = 0.5, color_ruler: [0, 2], color: #EE1271, stroke_ruler: [10]
    vega1_5 = vega_weighted_pointmap(200, 200, [-73.998427, 40.730309, -73.954348, 40.780816], ["#EE1271"], [0, 2],
                                     [10], 0.5, "EPSG:4326")
    baseline1_5 = weighted_pointmap(vega1_5, res1)
    weighted_point_map1_5_1 = weighted_pointmap(vega1_5, res1)
    weighted_point_map1_5_2 = weighted_pointmap(vega1_5, res1)

    baseline_png1_5 = png_path + "weighted_point_map_nyc_1_5.png"
    save_png(baseline1_5, baseline_png1_5)
    save_png(weighted_point_map1_5_1, png_path + "test_weighted_point_map_nyc_1_5-1.png")
    save_png(weighted_point_map1_5_2, png_path + "test_weighted_point_map_nyc_1_5-2.png")

    # 2 multiple color; single point size
    res2 = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    # 2.1 opacity = 1.0, color_ruler: [0, 2], color: red_transparency
    vega2_1 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#FF0000", "#FF0000"],
                                     [0, 2], [10], 1.0, "EPSG:4326")
    baseline2_1 = weighted_pointmap(vega2_1, res2)
    weighted_point_map2_1_1 = weighted_pointmap(vega2_1, res2)
    weighted_point_map2_1_2 = weighted_pointmap(vega2_1, res2)

    baseline_png2_1 = png_path + "weighted_point_map_nyc_2_1.png"
    save_png(baseline2_1, baseline_png2_1)
    save_png(weighted_point_map2_1_1, png_path + "test_weighted_point_map_nyc_2_1-1.png")
    save_png(weighted_point_map2_1_2, png_path + "test_weighted_point_map_nyc_2_1-2.png")

    # 2.2 opacity = 0.0, color_ruler: [1, 10]
    vega2_2 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#FF00FF", "#FFFF00"],
                                     [1, 10], [6], 0.0, "EPSG:4326")
    baseline2_2 = weighted_pointmap(vega2_2, res2)
    weighted_point_map2_2_1 = weighted_pointmap(vega2_2, res2)
    weighted_point_map2_2_2 = weighted_pointmap(vega2_2, res2)

    baseline_png2_2 = png_path + "weighted_point_map_nyc_2_2.png"
    save_png(baseline2_2, baseline_png2_2)
    save_png(weighted_point_map2_2_1, png_path + "test_weighted_point_map_nyc_2_2-1.png")
    save_png(weighted_point_map2_2_2, png_path + "test_weighted_point_map_nyc_2_2-2.png")

    # 2.3 opacity = 0.5, color_ruler: [0, 100]
    vega2_3 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#0000FF", "#0000FF"],
                                     [0, 100], [5], 0.5, "EPSG:4326")
    baseline2_3 = weighted_pointmap(vega2_3, res2)
    weighted_point_map2_3_1 = weighted_pointmap(vega2_3, res2)
    weighted_point_map2_3_2 = weighted_pointmap(vega2_3, res2)

    baseline_png2_3 = png_path + "weighted_point_map_nyc_2_3.png"
    save_png(baseline2_3, baseline_png2_3)
    save_png(weighted_point_map2_3_1, png_path + "test_weighted_point_map_nyc_2_3-1.png")
    save_png(weighted_point_map2_3_2, png_path + "test_weighted_point_map_nyc_2_3-2.png")

    # 2.4 opacity = 0.5, color_ruler: [0, 2], color: white_blue, stroke_ruler: [0]
    vega2_4 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#E2E2E2", "#115F9A"],
                                     [1, 10], [0], 0.5, "EPSG:4326")
    baseline2_4 = weighted_pointmap(vega2_4, res2)
    weighted_point_map2_4_1 = weighted_pointmap(vega2_4, res2)
    weighted_point_map2_4_2 = weighted_pointmap(vega2_4, res2)

    baseline_png2_4 = png_path + "weighted_point_map_nyc_2_4.png"
    save_png(baseline2_4, baseline_png2_4)
    save_png(weighted_point_map2_4_1, png_path + "test_weighted_point_map_nyc_2_4-1.png")
    save_png(weighted_point_map2_4_2, png_path + "test_weighted_point_map_nyc_2_4-2.png")

    # 2.5 size: 200*200, opacity = 1.0, color_ruler: [0, 2], color: green_yellow_red, stroke_ruler: [1]
    vega2_5 = vega_weighted_pointmap(200, 200, [-73.998427, 40.730309, -73.954348, 40.780816], ["#4D904F", "#C23728"],
                                     [0, 2], [1], 1.0, "EPSG:4326")
    baseline2_5 = weighted_pointmap(vega2_5, res2)
    weighted_point_map2_5_1 = weighted_pointmap(vega2_5, res2)
    weighted_point_map2_5_2 = weighted_pointmap(vega2_5, res2)

    baseline_png2_5 = png_path + "weighted_point_map_nyc_2_5.png"
    save_png(baseline2_5, baseline_png2_5)
    save_png(weighted_point_map2_5_1, png_path + "test_weighted_point_map_nyc_2_5-1.png")
    save_png(weighted_point_map2_5_2, png_path + "test_weighted_point_map_nyc_2_5-2.png")

    # 3 single color; multiple point size
    res3 = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point, fare_amount as s from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    # 3.1 opacity = 1.0, color_ruler: [0, 2], color: #900E46(red)
    vega3_1 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#900E46"], [0, 2],
                                     [0, 10], 1.0, "EPSG:4326")
    baseline3_1 = weighted_pointmap(vega3_1, res3)
    weighted_point_map3_1_1 = weighted_pointmap(vega3_1, res3)
    weighted_point_map3_1_2 = weighted_pointmap(vega3_1, res3)

    baseline_png3_1 = png_path + "weighted_point_map_nyc_3_1.png"
    save_png(baseline3_1, baseline_png3_1)
    save_png(weighted_point_map3_1_1, png_path + "test_weighted_point_map_nyc_3_1-1.png")
    save_png(weighted_point_map3_1_2, png_path + "test_weighted_point_map_nyc_3_1-2.png")

    # 3.2 opacity = 0.0, color_ruler: [1, 10], color: #4A4145(black)
    vega3_2 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#4A4145"], [1, 10],
                                     [0, 10], 0.0, "EPSG:4326")
    baseline3_2 = weighted_pointmap(vega3_2, res3)
    weighted_point_map3_2_1 = weighted_pointmap(vega3_2, res3)
    weighted_point_map3_2_2 = weighted_pointmap(vega3_2, res3)

    baseline_png3_2 = png_path + "weighted_point_map_nyc_3_2.png"
    save_png(baseline3_2, baseline_png3_2)
    save_png(weighted_point_map3_2_1, png_path + "test_weighted_point_map_nyc_3_2-1.png")
    save_png(weighted_point_map3_2_2, png_path + "test_weighted_point_map_nyc_3_2-2.png")

    # 3.3 opacity = 0.5, color_ruler: [0, 100], color: #4A4145(black)
    vega3_3 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#4A4145"], [0, 100],
                                     [2, 8], 0.5, "EPSG:4326")
    baseline3_3 = weighted_pointmap(vega3_3, res3)
    weighted_point_map3_3_1 = weighted_pointmap(vega3_3, res3)
    weighted_point_map3_3_2 = weighted_pointmap(vega3_3, res3)

    baseline_png3_3 = png_path + "weighted_point_map_nyc_3_3.png"
    save_png(baseline3_3, baseline_png3_3)
    save_png(weighted_point_map3_3_1, png_path + "test_weighted_point_map_nyc_3_3-1.png")
    save_png(weighted_point_map3_3_2, png_path + "test_weighted_point_map_nyc_3_3-2.png")

    # 3.4 opacity = 0.5, color_ruler: [0, 2], color: #3574F0(blue), stroke_ruler: [1, 20]
    vega3_4 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#3574F0"], [0, 2],
                                     [1, 20], 0.5, "EPSG:4326")
    baseline3_4 = weighted_pointmap(vega3_4, res3)
    weighted_point_map3_4_1 = weighted_pointmap(vega3_4, res3)
    weighted_point_map3_4_2 = weighted_pointmap(vega3_4, res3)

    baseline_png3_4 = png_path + "weighted_point_map_nyc_3_4.png"
    save_png(baseline3_4, baseline_png3_4)
    save_png(weighted_point_map3_4_1, png_path + "test_weighted_point_map_nyc_3_4-1.png")
    save_png(weighted_point_map3_4_2, png_path + "test_weighted_point_map_nyc_3_4-2.png")

    # 3.5 size: 200*200, opacity = 1.0, color_ruler: [0, 2], color: #14EE47(green), stroke_ruler: [5, 11]
    vega3_5 = vega_weighted_pointmap(200, 200, [-73.998427, 40.730309, -73.954348, 40.780816], ["#14EE47"], [0, 2],
                                     [5, 11], 1.0, "EPSG:4326")
    baseline3_5 = weighted_pointmap(vega3_5, res3)
    weighted_point_map3_5_1 = weighted_pointmap(vega3_5, res3)
    weighted_point_map3_5_2 = weighted_pointmap(vega3_5, res3)

    baseline_png3_5 = png_path + "weighted_point_map_nyc_3_5.png"
    save_png(baseline3_5, baseline_png3_5)
    save_png(weighted_point_map3_5_1, png_path + "test_weighted_point_map_nyc_3_5-1.png")
    save_png(weighted_point_map3_5_2, png_path + "test_weighted_point_map_nyc_3_5-2.png")

    # 4 multiple color; multiple point size
    res4 = spark.sql("select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c, fare_amount as s from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    # 4.1 opacity = 1.0, color_ruler: [0, 2], color: green_yellow_red
    vega4_1 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#4D904F", "#C23728"], [0, 2],
                                     [0, 10], 1.0, "EPSG:4326")
    baseline4_1 = weighted_pointmap(vega4_1, res4)
    weighted_point_map4_1_1 = weighted_pointmap(vega4_1, res4)
    weighted_point_map4_1_2 = weighted_pointmap(vega4_1, res4)

    baseline_png4_1 = png_path + "weighted_point_map_nyc_4_1.png"
    save_png(baseline4_1, baseline_png4_1)
    save_png(weighted_point_map4_1_1, png_path + "test_weighted_point_map_nyc_4_1-1.png")
    save_png(weighted_point_map4_1_2, png_path + "test_weighted_point_map_nyc_4_1-2.png")

    # 4.2 opacity = 0.0, color_ruler: [1, 10]
    vega4_2 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#B4E7F5", "#FFFFFF"],
                                     [1, 10], [0, 10], 0.0, "EPSG:4326")
    baseline4_2 = weighted_pointmap(vega4_2, res4)
    weighted_point_map4_2_1 = weighted_pointmap(vega4_2, res4)
    weighted_point_map4_2_2 = weighted_pointmap(vega4_2, res4)

    baseline_png4_2 = png_path + "weighted_point_map_nyc_4_2.png"
    save_png(baseline4_2, baseline_png4_2)
    save_png(weighted_point_map4_2_1, png_path + "test_weighted_point_map_nyc_4_2-1.png")
    save_png(weighted_point_map4_2_2, png_path + "test_weighted_point_map_nyc_4_2-2.png")

    # 4.3 opacity = 0.5, color_ruler: [1, 5], color: blue_green_yellow
    vega4_3 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#115F9A", "#D0F401"],
                                     [1, 5], [0, 10], 0.5, "EPSG:4326")
    baseline4_3 = weighted_pointmap(vega4_3, res4)
    weighted_point_map4_3_1 = weighted_pointmap(vega4_3, res4)
    weighted_point_map4_3_2 = weighted_pointmap(vega4_3, res4)

    baseline_png4_3 = png_path + "weighted_point_map_nyc_4_3.png"
    save_png(baseline4_3, baseline_png4_3)
    save_png(weighted_point_map4_3_1, png_path + "test_weighted_point_map_nyc_4_3-1.png")
    save_png(weighted_point_map4_3_2, png_path + "test_weighted_point_map_nyc_4_3-2.png")

    # 4.4 opacity = 0.5, color_ruler: [0, 5], color: blue_green_yellow, stroke_ruler: [1, 11]
    vega4_4 = vega_weighted_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], ["#115F9A", "#D0F401"],
                                     [0, 5], [1, 11], 0.5, "EPSG:4326")
    baseline4_4 = weighted_pointmap(vega4_4, res4)
    weighted_point_map4_4_1 = weighted_pointmap(vega4_4, res4)
    weighted_point_map4_4_2 = weighted_pointmap(vega4_4, res4)

    baseline_png4_4 = png_path + "weighted_point_map_nyc_4_4.png"
    save_png(baseline4_4, baseline_png4_4)
    save_png(weighted_point_map4_4_1, png_path + "test_weighted_point_map_nyc_4_4-1.png")
    save_png(weighted_point_map4_4_2, png_path + "test_weighted_point_map_nyc_4_4-2.png")

    # 4.5 size: 200*200, opacity = 1.0, color_ruler: [0, 2], color: blue_transparency, stroke_ruler: [5, 15]
    vega4_5 = vega_weighted_pointmap(200, 200, [-73.998427, 40.730309, -73.954348, 40.780816], ["#0000FF", "#0000FF"],
                                     [0, 2], [5, 15], 1.0, "EPSG:4326")
    baseline4_5 = weighted_pointmap(vega4_5, res4)
    weighted_point_map4_5_1 = weighted_pointmap(vega4_5, res4)
    weighted_point_map4_5_2 = weighted_pointmap(vega4_5, res4)

    baseline_png4_5 = png_path + "weighted_point_map_nyc_4_5.png"
    save_png(baseline4_5, baseline_png4_5)
    save_png(weighted_point_map4_5_1, png_path + "test_weighted_point_map_nyc_4_5-1.png")
    save_png(weighted_point_map4_5_2, png_path + "test_weighted_point_map_nyc_4_5-2.png")

    spark.catalog.dropGlobalTempView("nyc_taxi")

    assert run_diff_png(baseline_png1_1, png_path + "test_weighted_point_map_nyc_1_1-1.png")
    assert run_diff_png(baseline_png1_1, png_path + "test_weighted_point_map_nyc_1_1-2.png")
    assert run_diff_png(baseline_png1_2, png_path + "test_weighted_point_map_nyc_1_2-1.png")
    assert run_diff_png(baseline_png1_2, png_path + "test_weighted_point_map_nyc_1_2-2.png")
    assert run_diff_png(baseline_png1_3, png_path + "test_weighted_point_map_nyc_1_3-1.png")
    assert run_diff_png(baseline_png1_3, png_path + "test_weighted_point_map_nyc_1_3-2.png")
    assert run_diff_png(baseline_png1_4, png_path + "test_weighted_point_map_nyc_1_4-1.png")
    assert run_diff_png(baseline_png1_4, png_path + "test_weighted_point_map_nyc_1_4-2.png")
    assert run_diff_png(baseline_png1_5, png_path + "test_weighted_point_map_nyc_1_5-1.png")
    assert run_diff_png(baseline_png1_5, png_path + "test_weighted_point_map_nyc_1_5-2.png")
    assert run_diff_png(baseline_png2_1, png_path + "test_weighted_point_map_nyc_2_1-1.png")
    assert run_diff_png(baseline_png2_1, png_path + "test_weighted_point_map_nyc_2_1-2.png")
    assert run_diff_png(baseline_png2_2, png_path + "test_weighted_point_map_nyc_2_2-1.png")
    assert run_diff_png(baseline_png2_2, png_path + "test_weighted_point_map_nyc_2_2-2.png")
    assert run_diff_png(baseline_png2_3, png_path + "test_weighted_point_map_nyc_2_3-1.png")
    assert run_diff_png(baseline_png2_3, png_path + "test_weighted_point_map_nyc_2_3-2.png")
    assert run_diff_png(baseline_png2_4, png_path + "test_weighted_point_map_nyc_2_4-1.png")
    assert run_diff_png(baseline_png2_4, png_path + "test_weighted_point_map_nyc_2_4-2.png")
    assert run_diff_png(baseline_png2_5, png_path + "test_weighted_point_map_nyc_2_5-1.png")
    assert run_diff_png(baseline_png2_5, png_path + "test_weighted_point_map_nyc_2_5-2.png")
    assert run_diff_png(baseline_png3_1, png_path + "test_weighted_point_map_nyc_3_1-1.png")
    assert run_diff_png(baseline_png3_1, png_path + "test_weighted_point_map_nyc_3_1-2.png")
    assert run_diff_png(baseline_png3_2, png_path + "test_weighted_point_map_nyc_3_2-1.png")
    assert run_diff_png(baseline_png3_2, png_path + "test_weighted_point_map_nyc_3_2-2.png")
    assert run_diff_png(baseline_png3_3, png_path + "test_weighted_point_map_nyc_3_3-1.png")
    assert run_diff_png(baseline_png3_3, png_path + "test_weighted_point_map_nyc_3_3-2.png")
    assert run_diff_png(baseline_png3_4, png_path + "test_weighted_point_map_nyc_3_4-1.png")
    assert run_diff_png(baseline_png3_4, png_path + "test_weighted_point_map_nyc_3_4-2.png")
    assert run_diff_png(baseline_png3_5, png_path + "test_weighted_point_map_nyc_3_5-1.png")
    assert run_diff_png(baseline_png3_5, png_path + "test_weighted_point_map_nyc_3_5-2.png")
    assert run_diff_png(baseline_png4_1, png_path + "test_weighted_point_map_nyc_4_1-1.png")
    assert run_diff_png(baseline_png4_1, png_path + "test_weighted_point_map_nyc_4_1-2.png")
    assert run_diff_png(baseline_png4_2, png_path + "test_weighted_point_map_nyc_4_2-1.png")
    assert run_diff_png(baseline_png4_2, png_path + "test_weighted_point_map_nyc_4_2-2.png")
    assert run_diff_png(baseline_png4_3, png_path + "test_weighted_point_map_nyc_4_3-1.png")
    assert run_diff_png(baseline_png4_3, png_path + "test_weighted_point_map_nyc_4_3-2.png")
    assert run_diff_png(baseline_png4_4, png_path + "test_weighted_point_map_nyc_4_4-1.png")
    assert run_diff_png(baseline_png4_4, png_path + "test_weighted_point_map_nyc_4_4-2.png")
    assert run_diff_png(baseline_png4_5, png_path + "test_weighted_point_map_nyc_4_5-1.png")
    assert run_diff_png(baseline_png4_5, png_path + "test_weighted_point_map_nyc_4_5-2.png")

# pylint: disable=too-many-statements
def run_test_heat_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, "
        "trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, "
        "dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, "
        "buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        file_path).cache()
    df.createOrReplaceTempView("nyc_taxi")

    register_funcs(spark)
    res = spark.sql(
        "select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    # 1 size:1024*896, map_scale: 10.0
    vega_1 = vega_heatmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 10.0, 'EPSG:4326')
    baseline1 = heatmap(vega_1, res)
    heat_map1_1 = heatmap(vega_1, res)
    heat_map1_2 = heatmap(vega_1, res)

    baseline_png1 = png_path + "heat_map_nyc_1.png"
    save_png(baseline1, baseline_png1)
    save_png(heat_map1_1, png_path + "test_heat_map_nyc_1-1.png")
    save_png(heat_map1_2, png_path + "test_heat_map_nyc_1-2.png")

    # 2 map_scale: 0.0
    vega_2 = vega_heatmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 0.0, 'EPSG:4326')
    baseline2 = heatmap(vega_2, res)
    heat_map2_1 = heatmap(vega_2, res)
    heat_map2_2 = heatmap(vega_2, res)

    baseline_png2 = png_path + "heat_map_nyc_2.png"
    save_png(baseline2, baseline_png2)
    save_png(heat_map2_1, png_path + "test_heat_map_nyc_2-1.png")
    save_png(heat_map2_2, png_path + "test_heat_map_nyc_2-2.png")

    # 3 map_scale: 12.0
    vega_3 = vega_heatmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 12.0, 'EPSG:4326')
    baseline3 = heatmap(vega_3, res)
    heat_map3_1 = heatmap(vega_3, res)
    heat_map3_2 = heatmap(vega_3, res)

    baseline_png3 = png_path + "heat_map_nyc_3.png"
    save_png(baseline3, baseline_png3)
    save_png(heat_map3_1, png_path + "test_heat_map_nyc_3-1.png")
    save_png(heat_map3_2, png_path + "test_heat_map_nyc_3-2.png")

    # 4 map_scale: 5.5
    vega_4 = vega_heatmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 5.5, 'EPSG:4326')
    baseline4 = heatmap(vega_4, res)
    heat_map4_1 = heatmap(vega_4, res)
    heat_map4_2 = heatmap(vega_4, res)

    baseline_png4 = png_path + "heat_map_nyc_4.png"
    save_png(baseline4, baseline_png4)
    save_png(heat_map4_1, png_path + "test_heat_map_nyc_4-1.png")
    save_png(heat_map4_2, png_path + "test_heat_map_nyc_4-2.png")

    # 5 size:200*200
    vega_5 = vega_heatmap(200, 200, [-73.998427, 40.730309, -73.954348, 40.780816], 10.0, 'EPSG:4326')
    baseline5 = heatmap(vega_5, res)
    heat_map5_1 = heatmap(vega_5, res)
    heat_map5_2 = heatmap(vega_5, res)

    baseline_png5 = png_path + "heat_map_nyc_5.png"
    save_png(baseline5, baseline_png5)
    save_png(heat_map5_1, png_path + "test_heat_map_nyc_5-1.png")
    save_png(heat_map5_2, png_path + "test_heat_map_nyc_5-2.png")

    spark.catalog.dropGlobalTempView("nyc_taxi")

    assert run_diff_png(baseline_png1, png_path + "test_heat_map_nyc_1-1.png", 0.1)
    assert run_diff_png(baseline_png1, png_path + "test_heat_map_nyc_1-2.png", 0.1)
    assert run_diff_png(baseline_png2, png_path + "test_heat_map_nyc_2-1.png", 0.1)
    assert run_diff_png(baseline_png2, png_path + "test_heat_map_nyc_2-2.png", 0.1)
    assert run_diff_png(baseline_png3, png_path + "test_heat_map_nyc_3-1.png", 0.15)
    assert run_diff_png(baseline_png3, png_path + "test_heat_map_nyc_3-2.png", 0.15)
    assert run_diff_png(baseline_png4, png_path + "test_heat_map_nyc_4-1.png", 0.1)
    assert run_diff_png(baseline_png4, png_path + "test_heat_map_nyc_4-2.png", 0.1)
    assert run_diff_png(baseline_png5, png_path + "test_heat_map_nyc_5-1.png", 0.2)
    assert run_diff_png(baseline_png5, png_path + "test_heat_map_nyc_5-2.png", 0.2)

# pylint: disable=too-many-statements
def run_test_choropleth_map(spark):
    df = spark.read.format("csv").option("header", True).option("delimiter", ",").schema(
        "VendorID string, tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, passenger_count long, "
        "trip_distance double, pickup_longitude double, pickup_latitude double, dropoff_longitude double, "
        "dropoff_latitude double, fare_amount double, tip_amount double, total_amount double, buildingid_pickup long, "
        "buildingid_dropoff long, buildingtext_pickup string, buildingtext_dropoff string").load(
        file_path).cache()
    df.createOrReplaceTempView("nyc_taxi")

    res = spark.sql("select ST_GeomFromText(buildingtext_dropoff) as wkt, passenger_count as w from nyc_taxi")

    # 1-9 test color_gradient
    # 1 blue_to_red
    vega_1 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#0000FF", "#FF0000"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline1 = choroplethmap(vega_1, res)
    choropleth_map1_1 = choroplethmap(vega_1, res)
    choropleth_map1_2 = choroplethmap(vega_1, res)

    baseline_png1 = png_path + "choropleth_map_nyc_1.png"
    save_png(baseline1, baseline_png1)
    save_png(choropleth_map1_1, png_path + "test_choropleth_map_nyc_1-1.png")
    save_png(choropleth_map1_2, png_path + "test_choropleth_map_nyc_1-2.png")

    # 2 green_yellow_red
    vega_2 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#4D904F", "#C23728"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline2 = choroplethmap(vega_2, res)
    choropleth_map2_1 = choroplethmap(vega_2, res)
    choropleth_map2_2 = choroplethmap(vega_2, res)

    baseline_png2 = png_path + "choropleth_map_nyc_2.png"
    save_png(baseline2, baseline_png2)
    save_png(choropleth_map2_1, png_path + "test_choropleth_map_nyc_2-1.png")
    save_png(choropleth_map2_2, png_path + "test_choropleth_map_nyc_2-2.png")

    # 3 blue_white_red
    vega_3 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#1984C5", "#C23728"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline3 = choroplethmap(vega_3, res)
    choropleth_map3_1 = choroplethmap(vega_3, res)
    choropleth_map3_2 = choroplethmap(vega_3, res)

    baseline_png3 = png_path + "choropleth_map_nyc_3.png"
    save_png(baseline3, baseline_png3)
    save_png(choropleth_map3_1, png_path + "test_choropleth_map_nyc_3-1.png")
    save_png(choropleth_map3_2, png_path + "test_choropleth_map_nyc_3-2.png")

    # 4 skyblue_to_white
    vega_4 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#B4E7F5", "#FFFFFF"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline4 = choroplethmap(vega_4, res)
    choropleth_map4_1 = choroplethmap(vega_4, res)
    choropleth_map4_2 = choroplethmap(vega_4, res)

    baseline_png4 = png_path + "choropleth_map_nyc_4.png"
    save_png(baseline4, baseline_png4)
    save_png(choropleth_map4_1, png_path + "test_choropleth_map_nyc_4-1.png")
    save_png(choropleth_map4_2, png_path + "test_choropleth_map_nyc_4-2.png")

    # 5 purple_to_yellow
    vega_5 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF00FF", "#FFFF00"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline5 = choroplethmap(vega_5, res)
    choropleth_map5_1 = choroplethmap(vega_5, res)
    choropleth_map5_2 = choroplethmap(vega_5, res)

    baseline_png5 = png_path + "choropleth_map_nyc_5.png"
    save_png(baseline5, baseline_png5)
    save_png(choropleth_map5_1, png_path + "test_choropleth_map_nyc_5-1.png")
    save_png(choropleth_map5_2, png_path + "test_choropleth_map_nyc_5-2.png")

    # 6 red_transparency
    vega_6 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF0000", "#FF0000"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline6 = choroplethmap(vega_6, res)
    choropleth_map6_1 = choroplethmap(vega_6, res)
    choropleth_map6_2 = choroplethmap(vega_6, res)

    baseline_png6 = png_path + "choropleth_map_nyc_6.png"
    save_png(baseline6, baseline_png6)
    save_png(choropleth_map6_1, png_path + "test_choropleth_map_nyc_6-1.png")
    save_png(choropleth_map6_2, png_path + "test_choropleth_map_nyc_6-2.png")

    # 7 blue_transparency
    vega_7 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#0000FF", "0000FF"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline7 = choroplethmap(vega_7, res)
    choropleth_map7_1 = choroplethmap(vega_7, res)
    choropleth_map7_2 = choroplethmap(vega_7, res)

    baseline_png7 = png_path + "choropleth_map_nyc_7.png"
    save_png(baseline7, baseline_png7)
    save_png(choropleth_map7_1, png_path + "test_choropleth_map_nyc_7-1.png")
    save_png(choropleth_map7_2, png_path + "test_choropleth_map_nyc_7-2.png")

    # 8 blue_green_yellow
    vega_8 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#115F9A", "#D0F401"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline8 = choroplethmap(vega_8, res)
    choropleth_map8_1 = choroplethmap(vega_8, res)
    choropleth_map8_2 = choroplethmap(vega_8, res)

    baseline_png8 = png_path + "choropleth_map_nyc_8.png"
    save_png(baseline8, baseline_png8)
    save_png(choropleth_map8_1, png_path + "test_choropleth_map_nyc_8-1.png")
    save_png(choropleth_map8_2, png_path + "test_choropleth_map_nyc_8-2.png")

    # 9 white_blue
    vega_9 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#E2E2E2", "#115F9A"],
                                [2.5, 5], 1.0, 'EPSG:4326')
    baseline9 = choroplethmap(vega_9, res)
    choropleth_map9_1 = choroplethmap(vega_9, res)
    choropleth_map9_2 = choroplethmap(vega_9, res)

    baseline_png9 = png_path + "choropleth_map_nyc_9.png"
    save_png(baseline9, baseline_png9)
    save_png(choropleth_map9_1, png_path + "test_choropleth_map_nyc_9-1.png")
    save_png(choropleth_map9_2, png_path + "test_choropleth_map_nyc_9-2.png")

    # 10-12 test ruler
    # 10 ruler: [1, 500]
    vega_10 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#0000FF", "#FF0000"],
                                 [1, 500], 1.0, 'EPSG:4326')
    baseline10 = choroplethmap(vega_10, res)
    choropleth_map10_1 = choroplethmap(vega_10, res)
    choropleth_map10_2 = choroplethmap(vega_10, res)

    baseline_png10 = png_path + "choropleth_map_nyc_10.png"
    save_png(baseline10, baseline_png10)
    save_png(choropleth_map10_1, png_path + "test_choropleth_map_nyc_10-1.png")
    save_png(choropleth_map10_2, png_path + "test_choropleth_map_nyc_10-2.png")

    # 11 ruler: [1, 10000]
    vega_11 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#0000FF", "#FF0000"],
                                 [1, 10000], 1.0, 'EPSG:4326')
    baseline11 = choroplethmap(vega_11, res)
    choropleth_map11_1 = choroplethmap(vega_11, res)
    choropleth_map11_2 = choroplethmap(vega_11, res)

    baseline_png11 = png_path + "choropleth_map_nyc_11.png"
    save_png(baseline11, baseline_png11)
    save_png(choropleth_map11_1, png_path + "test_choropleth_map_nyc_11-1.png")
    save_png(choropleth_map11_2, png_path + "test_choropleth_map_nyc_11-2.png")

    # 12 ruler: [0, 2.5]
    vega_12 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#0000FF", "#FF0000"],
                                 [0, 2.5], 1.0, 'EPSG:4326')
    baseline12 = choroplethmap(vega_12, res)
    choropleth_map12_1 = choroplethmap(vega_12, res)
    choropleth_map12_2 = choroplethmap(vega_12, res)

    baseline_png12 = png_path + "choropleth_map_nyc_12.png"
    save_png(baseline12, baseline_png12)
    save_png(choropleth_map12_1, png_path + "test_choropleth_map_nyc_12-1.png")
    save_png(choropleth_map12_2, png_path + "test_choropleth_map_nyc_12-2.png")

    # 13-15 test opacity
    # 13 opacity: 0.0
    vega_13 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF00FF", "#FFFF00"],
                                 [2.5, 5], 0.0, 'EPSG:4326')
    baseline13 = choroplethmap(vega_13, res)
    choropleth_map13_1 = choroplethmap(vega_13, res)
    choropleth_map13_2 = choroplethmap(vega_13, res)

    baseline_png13 = png_path + "choropleth_map_nyc_13.png"
    save_png(baseline13, baseline_png13)
    save_png(choropleth_map13_1, png_path + "test_choropleth_map_nyc_13-1.png")
    save_png(choropleth_map13_2, png_path + "test_choropleth_map_nyc_13-2.png")

    # 14 opacity: 1.0
    vega_14 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF00FF", "#FFFF00"],
                                 [2.5, 5], 1.0, 'EPSG:4326')
    baseline14 = choroplethmap(vega_14, res)
    choropleth_map14_1 = choroplethmap(vega_14, res)
    choropleth_map14_2 = choroplethmap(vega_14, res)

    baseline_png14 = png_path + "choropleth_map_nyc_14.png"
    save_png(baseline14, baseline_png14)
    save_png(choropleth_map14_1, png_path + "test_choropleth_map_nyc_14-1.png")
    save_png(choropleth_map14_2, png_path + "test_choropleth_map_nyc_14-2.png")

    # 15 opacity: 0.5
    vega_15 = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF00FF", "#FFFF00"],
                                 [2.5, 5], 0.5, 'EPSG:4326')
    baseline15 = choroplethmap(vega_15, res)
    choropleth_map15_1 = choroplethmap(vega_15, res)
    choropleth_map15_2 = choroplethmap(vega_15, res)

    baseline_png15 = png_path + "choropleth_map_nyc_15.png"
    save_png(baseline15, baseline_png15)
    save_png(choropleth_map15_1, png_path + "test_choropleth_map_nyc_15-1.png")
    save_png(choropleth_map15_2, png_path + "test_choropleth_map_nyc_15-2.png")

    # 16-18 test size
    # 16 width: 256, height: 256
    vega_16 = vega_choroplethmap(256, 256, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF00FF", "#FFFF00"],
                                 [2.5, 5], 1.0, 'EPSG:4326')
    baseline16 = choroplethmap(vega_16, res)
    choropleth_map16_1 = choroplethmap(vega_16, res)
    choropleth_map16_2 = choroplethmap(vega_16, res)

    baseline_png16 = png_path + "choropleth_map_nyc_16.png"
    save_png(baseline16, baseline_png16)
    save_png(choropleth_map16_1, png_path + "test_choropleth_map_nyc_16-1.png")
    save_png(choropleth_map16_2, png_path + "test_choropleth_map_nyc_16-2.png")

    # 17 width: 200, height: 200
    vega_17 = vega_choroplethmap(200, 200, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF00FF", "#FFFF00"],
                                 [2.5, 5], 1.0, 'EPSG:4326')
    baseline17 = choroplethmap(vega_17, res)
    choropleth_map17_1 = choroplethmap(vega_17, res)
    choropleth_map17_2 = choroplethmap(vega_17, res)

    baseline_png17 = png_path + "choropleth_map_nyc_17.png"
    save_png(baseline17, baseline_png17)
    save_png(choropleth_map17_1, png_path + "test_choropleth_map_nyc_17-1.png")
    save_png(choropleth_map17_2, png_path + "test_choropleth_map_nyc_17-2.png")

    # 18 width: 500, height: 200
    vega_18 = vega_choroplethmap(500, 200, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF00FF", "#FFFF00"],
                                 [2.5, 5], 1.0, 'EPSG:4326')
    baseline18 = choroplethmap(vega_18, res)
    choropleth_map18_1 = choroplethmap(vega_18, res)
    choropleth_map18_2 = choroplethmap(vega_18, res)

    baseline_png18 = png_path + "choropleth_map_nyc_18.png"
    save_png(baseline18, baseline_png18)
    save_png(choropleth_map18_1, png_path + "test_choropleth_map_nyc_18-1.png")
    save_png(choropleth_map18_2, png_path + "test_choropleth_map_nyc_18-2.png")

    # 19 width: 10, height: 10
    vega_19 = vega_choroplethmap(10, 10, [-73.994092, 40.753893, -73.977588, 40.759642], ["#FF00FF", "#FFFF00"],
                                 [2.5, 5], 1.0, 'EPSG:4326')
    baseline19 = choroplethmap(vega_19, res)
    choropleth_map19_1 = choroplethmap(vega_19, res)
    choropleth_map19_2 = choroplethmap(vega_19, res)

    baseline_png19 = png_path + "choropleth_map_nyc_19.png"
    save_png(baseline19, baseline_png19)
    save_png(choropleth_map19_1, png_path + "test_choropleth_map_nyc_19-1.png")
    save_png(choropleth_map19_2, png_path + "test_choropleth_map_nyc_19-2.png")

    spark.catalog.dropGlobalTempView("nyc_taxi")

    assert run_diff_png(baseline_png1, png_path + "test_choropleth_map_nyc_1-1.png")
    assert run_diff_png(baseline_png1, png_path + "test_choropleth_map_nyc_1-2.png")
    assert run_diff_png(baseline_png2, png_path + "test_choropleth_map_nyc_2-1.png")
    assert run_diff_png(baseline_png2, png_path + "test_choropleth_map_nyc_2-2.png")
    assert run_diff_png(baseline_png3, png_path + "test_choropleth_map_nyc_3-1.png")
    assert run_diff_png(baseline_png3, png_path + "test_choropleth_map_nyc_3-2.png")
    assert run_diff_png(baseline_png4, png_path + "test_choropleth_map_nyc_4-1.png")
    assert run_diff_png(baseline_png4, png_path + "test_choropleth_map_nyc_4-2.png")
    assert run_diff_png(baseline_png5, png_path + "test_choropleth_map_nyc_5-1.png")
    assert run_diff_png(baseline_png5, png_path + "test_choropleth_map_nyc_5-2.png")
    assert run_diff_png(baseline_png6, png_path + "test_choropleth_map_nyc_6-1.png")
    assert run_diff_png(baseline_png6, png_path + "test_choropleth_map_nyc_6-2.png")
    assert run_diff_png(baseline_png7, png_path + "test_choropleth_map_nyc_7-1.png")
    assert run_diff_png(baseline_png7, png_path + "test_choropleth_map_nyc_7-2.png")
    assert run_diff_png(baseline_png8, png_path + "test_choropleth_map_nyc_8-1.png")
    assert run_diff_png(baseline_png8, png_path + "test_choropleth_map_nyc_8-2.png")
    assert run_diff_png(baseline_png9, png_path + "test_choropleth_map_nyc_9-1.png")
    assert run_diff_png(baseline_png9, png_path + "test_choropleth_map_nyc_9-2.png")
    assert run_diff_png(baseline_png10, png_path + "test_choropleth_map_nyc_10-1.png")
    assert run_diff_png(baseline_png10, png_path + "test_choropleth_map_nyc_10-2.png")
    assert run_diff_png(baseline_png11, png_path + "test_choropleth_map_nyc_11-1.png")
    assert run_diff_png(baseline_png11, png_path + "test_choropleth_map_nyc_11-2.png")
    assert run_diff_png(baseline_png12, png_path + "test_choropleth_map_nyc_12-1.png")
    assert run_diff_png(baseline_png12, png_path + "test_choropleth_map_nyc_12-2.png")
    assert run_diff_png(baseline_png13, png_path + "test_choropleth_map_nyc_13-1.png")
    assert run_diff_png(baseline_png13, png_path + "test_choropleth_map_nyc_13-2.png")
    assert run_diff_png(baseline_png14, png_path + "test_choropleth_map_nyc_14-1.png")
    assert run_diff_png(baseline_png14, png_path + "test_choropleth_map_nyc_14-2.png")
    assert run_diff_png(baseline_png15, png_path + "test_choropleth_map_nyc_15-1.png")
    assert run_diff_png(baseline_png15, png_path + "test_choropleth_map_nyc_15-2.png")
    assert run_diff_png(baseline_png16, png_path + "test_choropleth_map_nyc_16-1.png")
    assert run_diff_png(baseline_png16, png_path + "test_choropleth_map_nyc_16-2.png")
    assert run_diff_png(baseline_png17, png_path + "test_choropleth_map_nyc_17-1.png")
    assert run_diff_png(baseline_png17, png_path + "test_choropleth_map_nyc_17-2.png")
    assert run_diff_png(baseline_png18, png_path + "test_choropleth_map_nyc_18-1.png")
    assert run_diff_png(baseline_png18, png_path + "test_choropleth_map_nyc_18-2.png")
    assert run_diff_png(baseline_png19, png_path + "test_choropleth_map_nyc_19-1.png")
    assert run_diff_png(baseline_png19, png_path + "test_choropleth_map_nyc_19-2.png")


def run_test_icon_viz(spark):
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
        "select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))")

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    draw_png_path = dir_path + "/draw_map/taxi.png"

    # 1 size:1024*896
    vega_1 = vega_icon(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], draw_png_path, "EPSG:4326")
    baseline1 = icon_viz(vega_1, res)
    icon_viz1_1 = icon_viz(vega_1, res)
    icon_viz1_2 = icon_viz(vega_1, res)

    baseline_png1 = png_path + "icon_viz_nyc_1.png"
    save_png(baseline1, baseline_png1)
    save_png(icon_viz1_1, png_path + "test_icon_viz_nyc_1-1.png")
    save_png(icon_viz1_2, png_path + "test_icon_viz_nyc_1-2.png")

    # 2 size:200*200
    vega_2 = vega_icon(200, 200, [-73.998427, 40.730309, -73.954348, 40.780816], draw_png_path, "EPSG:4326")
    baseline2 = icon_viz(vega_2, res)
    icon_viz2_1 = icon_viz(vega_2, res)
    icon_viz2_2 = icon_viz(vega_2, res)

    baseline_png2 = png_path + "icon_viz_nyc_2.png"
    save_png(baseline2, baseline_png2)
    save_png(icon_viz2_1, png_path + "test_icon_viz_nyc_2-1.png")
    save_png(icon_viz2_2, png_path + "test_icon_viz_nyc_2-2.png")

    spark.catalog.dropGlobalTempView("nyc_taxi")

    assert run_diff_png(baseline_png1, png_path + "test_icon_viz_nyc_1-1.png")
    assert run_diff_png(baseline_png1, png_path + "test_icon_viz_nyc_1-2.png")
    assert run_diff_png(baseline_png2, png_path + "test_icon_viz_nyc_2-1.png")
    assert run_diff_png(baseline_png2, png_path + "test_icon_viz_nyc_2-2.png")


if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python Testmap") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_test_point_map(spark_session)
    run_test_weighted_point_map(spark_session)
    run_test_heat_map(spark_session)
    run_test_choropleth_map(spark_session)
    run_test_icon_viz(spark_session)

    spark_session.stop()
