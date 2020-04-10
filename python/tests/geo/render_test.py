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

import pandas
import arctern

from arctern.util import save_png
from arctern.util.vega import vega_pointmap, vega_weighted_pointmap, vega_heatmap, vega_choroplethmap


def test_projection():
    wkt = ["POINT (-8235193.62386326 4976211.44428777)"]
    top_left = "POINT (-8235871.4482427 4976468.32320551)"
    bottom_right = "POINT (-8235147.42627458 4976108.43009739)"

    arr_wkb = pandas.Series(arctern.wkt2wkb(wkt))
    arctern.projection(arr_wkb, bottom_right, top_left, 200, 300)


def test_transfrom_and_projection():
    wkt = ["POINT (-73.978003 40.754594)"]
    top_left = "POINT (-73.984092 40.756342)"
    bottom_right = "POINT (-73.977588 40.753893)"
    src_ts = "EPSG:4326"
    dst_rs = "EPSG:3857"

    arr_wkb = pandas.Series(arctern.wkt2wkb(wkt))
    arctern.transform_and_projection(arr_wkb, src_ts, dst_rs, bottom_right, top_left, 200, 300)


def test_point_map():
    x_data = []
    y_data = []

    # y = 150
    for i in range(100, 200):
        x_data.append(i)
        y_data.append(150)

    # y = x - 50
    for i in range(100, 200):
        x_data.append(i)
        y_data.append(i - 50)

    # y = 50
    for i in range(100, 200):
        x_data.append(i)
        y_data.append(50)

    arr_x = pandas.Series(x_data)
    arr_y = pandas.Series(y_data)
    points = arctern.ST_Point(arr_x, arr_y)

    vega = vega_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], point_size=3, point_color="#2DEF4A", opacity=0.5, coordinate_system="EPSG:4326")
    curve_z1 = arctern.point_map(vega, points)
    save_png(curve_z1, "/tmp/test_curve_z1.png")


def test_weighted_point_map():
    x_data = []
    y_data = []
    c_data = []
    s_data = []

    x_data.append(10)
    x_data.append(20)
    x_data.append(30)
    x_data.append(40)
    x_data.append(50)

    y_data.append(10)
    y_data.append(20)
    y_data.append(30)
    y_data.append(40)
    y_data.append(50)

    c_data.append(1)
    c_data.append(2)
    c_data.append(3)
    c_data.append(4)
    c_data.append(5)

    s_data.append(2)
    s_data.append(4)
    s_data.append(6)
    s_data.append(8)
    s_data.append(10)

    arr_x = pandas.Series(x_data)
    arr_y = pandas.Series(y_data)
    points = arctern.ST_Point(arr_x, arr_y)
    arr_c = pandas.Series(c_data)
    arr_s = pandas.Series(s_data)

    vega1 = vega_weighted_pointmap(300, 200, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#87CEEB"], opacity=1.0, coordinate_system="EPSG:3857")
    res1 = arctern.weighted_point_map(vega1, points)
    save_png(res1, "/tmp/test_weighted_0_0.png")

    vega2 = vega_weighted_pointmap(300, 200, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF", "#FF0000"], color_bound=[1, 5], opacity=1.0, coordinate_system="EPSG:3857")
    res2 = arctern.weighted_point_map(vega2, points, color_weights=arr_c)
    save_png(res2, "/tmp/test_weighted_1_0.png")

    vega3 = vega_weighted_pointmap(300, 200, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#87CEEB"], size_bound=[1, 10], opacity=1.0, coordinate_system="EPSG:3857")
    res3 = arctern.weighted_point_map(vega3, points, size_weights=arr_s)
    save_png(res3, "/tmp/test_weighted_0_1.png")

    vega4 = vega_weighted_pointmap(300, 200, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF", "#FF0000"], color_bound=[1, 5], size_bound=[1, 10], opacity=1.0, coordinate_system="EPSG:3857")
    res4 = arctern.weighted_point_map(vega4, points, color_weights=arr_c, size_weights=arr_s)
    save_png(res4, "/tmp/test_weighted_1_1.png")


def test_heat_map():
    x_data = []
    y_data = []
    c_data = []

    for i in range(0, 5):
        x_data.append(i + 50)
        y_data.append(i + 50)
        c_data.append(i + 50)

    arr_x = pandas.Series(x_data)
    arr_y = pandas.Series(y_data)
    points = arctern.ST_Point(arr_x, arr_y)
    arr_c = pandas.Series(y_data)

    vega = vega_heatmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], map_zoom_level=10.0, coordinate_system='EPSG:4326')
    heat_map1 = arctern.heat_map(vega, points, arr_c)

    save_png(heat_map1, "/tmp/test_heat_map1.png")


def test_choropleth_map():
    wkt_data = []
    count_data = []

    wkt_data.append("POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))")
    count_data.append(5.0)

    arr_wkt = pandas.Series(wkt_data)
    arr_wkb = arctern.wkt2wkb(arr_wkt)
    arr_count = pandas.Series(count_data)

    vega = vega_choroplethmap(1900, 1410, bounding_box=[-73.994092, 40.753893, -73.977588, 40.759642], color_gradient=["#0000FF", "#FF0000"], color_bound=[2.5, 5], opacity=1.0, coordinate_system='EPSG:4326')
    choropleth_map1 = arctern.choropleth_map(vega, arr_wkb, arr_count)
    save_png(choropleth_map1, "/tmp/test_choropleth_map1.png")
