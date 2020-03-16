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

import pyarrow
import pandas
import arctern

from arctern.util.vega.scatter_plot.vega_circle_2d import VegaCircle2d
from arctern.util.vega.heat_map.vega_heat_map import VegaHeatMap
from arctern.util.vega.choropleth_map.choropleth_map import VegaChoroplethMap

def _savePNG(data, png_name):
    try:
        imageData = data
    except BaseException as e:
        print(e)
    # save result png as fixed png
    else:
        with open(png_name, "wb") as tmp_file:
            tmp_file.write(imageData)

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

    vega_circle2d = VegaCircle2d(300, 200, 30, "#ff0000", 0.5)
    vega_json = vega_circle2d.build()

    curve_z = arctern.point_map(arr_x, arr_y, vega_json.encode('utf-8'))
    _savePNG(curve_z, "/tmp/curve_z.png")

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

    vega_heat_map = VegaHeatMap(300, 200, 10.0)
    vega_json = vega_heat_map.build()

    heat_map = arctern.heat_map(arr_x, arr_y, arr_c, vega_json.encode('utf-8'))
    _savePNG(heat_map, "/tmp/test_heat_map.png")

def test_choropleth_map():
    wkt_data = []
    count_data = []

    wkt_data.append("POLYGON (("
                    "-73.98128 40.754771, "
                    "-73.980185 40.754771, "
                    "-73.980185 40.755587, "
                    "-73.98128 40.755587, "
                    "-73.98128 40.754771))")
    count_data.append(5.0)

    arr_wkt = pandas.Series(wkt_data)
    arr_count = pandas.Series(count_data)

    vega_choropleth_map = VegaChoroplethMap(1900, 1410,
                                            [-73.984092, 40.753893, -73.977588, 40.756342],
                                            "blue_to_red", [2.5, 5], 1.0)
    vega_json = vega_choropleth_map.build()

    choropleth_map = arctern.choropleth_map(arr_wkt, arr_count, vega_json.encode('utf-8'))
    _savePNG(choropleth_map, "/tmp/test_choropleth_map.png")

test_choropleth_map()
