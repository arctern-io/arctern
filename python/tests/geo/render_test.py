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
from arctern.util.vega import vega_pointmap, vega_weighted_pointmap, vega_heatmap, vega_choroplethmap, vega_icon, vega_fishnetmap


def test_projection():
    point = arctern.ST_GeomFromText("POINT (-8235193.62386326 4976211.44428777)")[0]
    top_left = "POINT (-8235871.4482427 4976468.32320551)"
    bottom_right = "POINT (-8235147.42627458 4976108.43009739)"

    arr_wkb = pandas.Series([point])
    arctern.projection(arr_wkb, bottom_right, top_left, 200, 300)


def test_transfrom_and_projection():
    point = arctern.ST_GeomFromText("POINT (-73.978003 40.754594)")[0]
    top_left = "POINT (-73.984092 40.756342)"
    bottom_right = "POINT (-73.977588 40.753893)"
    src_ts = "EPSG:4326"
    dst_rs = "EPSG:3857"

    arr_wkb = pandas.Series([point])
    arctern.transform_and_projection(arr_wkb, src_ts, dst_rs, bottom_right, top_left, 200, 300)


def test_point_map():
    x_data = []
    y_data = []

    x_data.append(-73.96524)
    x_data.append(-73.96118)
    x_data.append(-73.97324)
    x_data.append(-73.98456)
    y_data.append(40.73747)
    y_data.append(40.74507)
    y_data.append(40.75890)
    y_data.append(40.77654)

    arr_x = pandas.Series(x_data)
    arr_y = pandas.Series(y_data)
    points = arctern.ST_Point(arr_x, arr_y)

    vega = vega_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], point_size=10, point_color="#0000FF", opacity=1.0, coordinate_system="EPSG:4326")
    curve_z1 = arctern.point_map_layer(vega, points)
    save_png(curve_z1, "/tmp/test_curve_z1.png")


def test_weighted_point_map():
    x_data = []
    y_data = []
    c_data = []
    s_data = []

    x_data.append(-73.96524)
    x_data.append(-73.96118)
    x_data.append(-73.97324)
    x_data.append(-73.98456)

    y_data.append(40.73747)
    y_data.append(40.74507)
    y_data.append(40.75890)
    y_data.append(40.77654)

    c_data.append(1)
    c_data.append(2)
    c_data.append(3)
    c_data.append(4)

    s_data.append(4)
    s_data.append(6)
    s_data.append(8)
    s_data.append(10)

    arr_x = pandas.Series(x_data)
    arr_y = pandas.Series(y_data)
    points = arctern.ST_Point(arr_x, arr_y)
    arr_c = pandas.Series(c_data)
    arr_s = pandas.Series(s_data)

    vega1 = vega_weighted_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF"], opacity=1.0, coordinate_system="EPSG:4326")
    res1 = arctern.weighted_point_map_layer(vega1, points)
    save_png(res1, "/tmp/test_weighted_0_0.png")

    vega2 = vega_weighted_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF", "#FF0000"], color_bound=[1, 5], opacity=1.0, coordinate_system="EPSG:4326")
    res2 = arctern.weighted_point_map_layer(vega2, points, color_weights=arr_c)
    save_png(res2, "/tmp/test_weighted_1_0.png")

    vega3 = vega_weighted_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF"], size_bound=[1, 10], opacity=1.0, coordinate_system="EPSG:4326")
    res3 = arctern.weighted_point_map_layer(vega3, points, size_weights=arr_s)
    save_png(res3, "/tmp/test_weighted_0_1.png")

    vega4 = vega_weighted_pointmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF", "#FF0000"], color_bound=[1, 5], size_bound=[1, 10], opacity=1.0, coordinate_system="EPSG:4326")
    res4 = arctern.weighted_point_map_layer(vega4, points, color_weights=arr_c, size_weights=arr_s)
    save_png(res4, "/tmp/test_weighted_1_1.png")

def test_heat_map():
    x_data = []
    y_data = []
    c_data = []

    x_data.append(-73.96524)
    x_data.append(-73.96118)
    x_data.append(-73.97324)
    x_data.append(-73.98456)

    y_data.append(40.73747)
    y_data.append(40.74507)
    y_data.append(40.75890)
    y_data.append(40.77654)

    c_data.append(10)
    c_data.append(20)
    c_data.append(30)
    c_data.append(40)

    arr_x = pandas.Series(x_data)
    arr_y = pandas.Series(y_data)
    arr_c = pandas.Series(c_data)
    points = arctern.ST_Point(arr_x, arr_y)

    vega = vega_heatmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], map_zoom_level=13.0, coordinate_system='EPSG:4326')
    heat_map1 = arctern.heat_map_layer(vega, points, arr_c)

    save_png(heat_map1, "/tmp/test_heat_map1.png")

def test_choropleth_map():
    wkt_data = []
    count_data = []

    wkt_data.append("POLYGON (("
      "-73.97324 40.73747, "
      "-73.96524 40.74507, "
      "-73.96118 40.75890, "
      "-73.95556 40.77654, "
      "-73.97324 40.73747))")
    count_data.append(5.0)

    arr_wkt = pandas.Series(wkt_data)
    arr_wkb = arctern.wkt2wkb(arr_wkt)
    arr_count = pandas.Series(count_data)

    vega = vega_choroplethmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF", "#FF0000"], color_bound=[2.5, 5], opacity=1.0, coordinate_system='EPSG:4326')
    choropleth_map1 = arctern.choropleth_map_layer(vega, arr_wkb, arr_count)
    save_png(choropleth_map1, "/tmp/test_choropleth_map1.png")


def test_icon_viz():
    x_data = []
    y_data = []

    x_data.append(-73.96524)
    x_data.append(-73.96118)
    x_data.append(-73.97324)
    x_data.append(-73.98456)

    y_data.append(40.73747)
    y_data.append(40.74507)
    y_data.append(40.75890)
    y_data.append(40.77654)

    arr_x = pandas.Series(x_data)
    arr_y = pandas.Series(y_data)
    points = arctern.ST_Point(arr_x, arr_y)

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    png_path = dir_path + "/../images/taxi.png"

    vega = vega_icon(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], icon_path=png_path, coordinate_system="EPSG:4326")

    icon_buf = arctern.icon_viz_layer(vega, points)
    save_png(icon_buf, "/tmp/test_icon_viz.png")

def test_fishnet_map():
    x_data = []
    y_data = []
    c_data = []

    x_data.append(-73.96524)
    x_data.append(-73.96118)
    x_data.append(-73.97324)
    x_data.append(-73.98456)

    y_data.append(40.73747)
    y_data.append(40.74507)
    y_data.append(40.75890)
    y_data.append(40.77654)

    c_data.append(10)
    c_data.append(20)
    c_data.append(30)
    c_data.append(40)

    arr_x = pandas.Series(x_data)
    arr_y = pandas.Series(y_data)
    arr_c = pandas.Series(c_data)
    points = arctern.ST_Point(arr_x, arr_y)

    vega = vega_fishnetmap(1024, 896, bounding_box=[-73.998427, 40.730309, -73.954348, 40.780816], color_gradient=["#0000FF", "#FF0000"], cell_size=4, cell_spacing=1, opacity=1.0, coordinate_system='EPSG:4326')
    heat_map1 = arctern.fishnet_map_layer(vega, points, arr_c)

    save_png(heat_map1, "/tmp/test_fishnetmap.png")
