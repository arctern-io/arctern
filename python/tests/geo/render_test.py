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

    vega_circle2d = vega_pointmap(1024, 896, [-73.998427, 40.730309, -73.954348, 40.780816], 3, "#2DEF4A", 0.5, "EPSG:43")
    vega_json = vega_circle2d.build()

    curve_z1 = arctern.point_map(arr_x, arr_y, vega_json.encode('utf-8'))
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
    arr_c = pandas.Series(c_data)
    arr_s = pandas.Series(s_data)

    vega = vega_weighted_pointmap(300, 200, [-73.998427, 40.730309, -73.954348, 40.780816], "blue_to_red", [1, 5], [1, 10], 1.0, "EPSG:3857")
    vega_json = vega.build()

    res1 = arctern.weighted_point_map(arr_x, arr_y, arr_c, arr_s, vega_json.encode('utf-8'))
    save_png(res1, "/tmp/test_weighted1.png")

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
    arr_c = pandas.Series(y_data)

    vega_heat_map = vega_heatmap(1024, 896, 10.0, [-73.998427, 40.730309, -73.954348, 40.780816], 'EPSG:4326')
    vega_json = vega_heat_map.build()

    heat_map1 = arctern.heat_map(arr_x, arr_y, arr_c, vega_json.encode('utf-8'))
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
    arr_count = pandas.Series(count_data)

    vega_choropleth_map = vega_choroplethmap(1900, 1410, [-73.994092, 40.753893, -73.977588, 40.759642], "blue_to_red", [2.5, 5], 1.0, 'EPSG:4326')
    vega_json = vega_choropleth_map.build()

    arr_wkb = arctern.wkt2wkb(arr_wkt)
    choropleth_map1 = arctern.choropleth_map(arr_wkb, arr_count, vega_json.encode('utf-8'))
    save_png(choropleth_map1, "/tmp/test_choropleth_map1.png")
