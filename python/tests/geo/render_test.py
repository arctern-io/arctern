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
import pandas
import arctern
import cv2

from arctern.util import save_png
from arctern.util.vega import vega_pointmap, vega_heatmap, vega_choroplethmap

map_path = sys.path[0] + "/../../../tests/expected/draw_map/"

def _diffPNG(baseline_png, compared_png, precision=0.00005):
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

            if is_point_equal == False:
                diff_point_num += 1

    if ((float)(diff_point_num) / (float)(baseline_size)) <= precision:
        return True
    else:
        return False

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

    vega_circle2d = vega_pointmap(1024, 896, 'POINT (-73.998427 40.780816)', 'POINT (-73.954348 40.730309)', 3, "#2DEF4A", 0.5, "EPSG:43")
    vega_json = vega_circle2d.build()

    curve_z1 = arctern.point_map(arr_x, arr_y, vega_json.encode('utf-8'))
    curve_z2 = arctern.point_map(arr_x, arr_y, vega_json.encode('utf-8'))
    curve_z3 = arctern.point_map(arr_x, arr_y, vega_json.encode('utf-8'))

    baseline_png = map_path + "curve_z.png"
    save_png(curve_z1, map_path + "test_curve_z1.png")
    save_png(curve_z2, map_path + "test_curve_z2.png")
    save_png(curve_z3, map_path + "test_curve_z3.png")

    assert _diffPNG(baseline_png, map_path + "test_curve_z1.png") == True
    assert _diffPNG(baseline_png, map_path + "test_curve_z2.png") == True
    assert _diffPNG(baseline_png, map_path + "test_curve_z3.png") == True

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

    vega_heat_map = vega_heatmap(1024, 896, 10.0, 'POINT (-73.998427 40.780816)', 'POINT (-73.954348 40.730309)', 'EPSG:4326')
    vega_json = vega_heat_map.build()

    baseline = arctern.heat_map(arr_x, arr_y, arr_c, vega_json.encode('utf-8'))
    heat_map1 = arctern.heat_map(arr_x, arr_y, arr_c, vega_json.encode('utf-8'))
    heat_map2 = arctern.heat_map(arr_x, arr_y, arr_c, vega_json.encode('utf-8'))
    heat_map3 = arctern.heat_map(arr_x, arr_y, arr_c, vega_json.encode('utf-8'))

    baseline_png = map_path + "heat_map.png"
    save_png(heat_map1, map_path + "test_heat_map1.png")
    save_png(heat_map2, map_path + "test_heat_map2.png")
    save_png(heat_map3, map_path + "test_heat_map3.png")

    assert _diffPNG(baseline_png, map_path + "test_heat_map1.png") == True
    assert _diffPNG(baseline_png, map_path + "test_heat_map2.png") == True
    assert _diffPNG(baseline_png, map_path + "test_heat_map3.png") == True

def test_choropleth_map():
    wkt_data = []
    count_data = []

    wkt_data.append("POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))");
    count_data.append(5.0)

    arr_wkt = pandas.Series(wkt_data)
    arr_count = pandas.Series(count_data)

    vega_choropleth_map = vega_choroplethmap(1900, 1410, 'POINT (-73.994092 40.759642)', 'POINT (-73.977588 40.753893)', "blue_to_red", [2.5, 5], 1.0, 'EPSG:4326')
    vega_json = vega_choropleth_map.build()

    baseline = arctern.choropleth_map(arr_wkt, arr_count, vega_json.encode('utf-8'))
    choropleth_map1 = arctern.choropleth_map(arr_wkt, arr_count, vega_json.encode('utf-8'))
    choropleth_map2 = arctern.choropleth_map(arr_wkt, arr_count, vega_json.encode('utf-8'))
    choropleth_map3 = arctern.choropleth_map(arr_wkt, arr_count, vega_json.encode('utf-8'))

    baseline_png = map_path + "choropleth_map.png"
    save_png(choropleth_map1, map_path + "test_choropleth_map1.png")
    save_png(choropleth_map2, map_path + "test_choropleth_map2.png")
    save_png(choropleth_map3, map_path + "test_choropleth_map3.png")

    assert _diffPNG(baseline_png, map_path + "test_choropleth_map1.png") == True
    assert _diffPNG(baseline_png, map_path + "test_choropleth_map2.png") == True
    assert _diffPNG(baseline_png, map_path + "test_choropleth_map3.png") == True
