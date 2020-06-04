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

# rename to arctern.plot later
# pylint: disable=reimported
import arctern as ap


def _get_matplot():
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(10, 6), dpi=200)
    return ax


def _finalize(ax, name):
    import matplotlib.pyplot as plt
    _ = ax
    plt.savefig("/tmp/plot_with_map_" + name + ".png")

# copied from source, since _func can not be called here
# pylint: disable=redefined-outer-name
def _transform_bbox(bounding_box, src_coord_sys, dst_coord_sys):
    import pyproj
    if src_coord_sys != dst_coord_sys:
        x0, y0, x1, y1 = bounding_box
        dst_proj = pyproj.Proj(dst_coord_sys)
        src_proj = pyproj.Proj(src_coord_sys)
        x0, y0 = pyproj.transform(src_proj, dst_proj, x0, y0, always_xy=True)
        x1, y1 = pyproj.transform(src_proj, dst_proj, x1, y1, always_xy=True)
        bounding_box = (x0, y0, x1, y1)
    return bounding_box

bounding_box = [-73.998427, 40.730309, -73.954348, 40.780816]

def test_contextily():
    import contextily as cx
    ax = _get_matplot()
    bbox = _transform_bbox(bounding_box, 'epsg:4326', 'epsg:3857')

    ax.set(xlim=(bbox[0], bbox[2]), ylim=(bbox[1], bbox[3]))
    cx.add_basemap(ax)
    _finalize(ax, "test_contextily")


def test_pointmap():
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
    ax = _get_matplot()
    ap.plot.pointmap(ax, points, bounding_box=bounding_box, point_size=10,
                     point_color="#0000FF", opacity=1.0, coordinate_system="EPSG:4326")
    _finalize(ax, "test_point_map")


def test_weighted_pointmap():
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

    ax = _get_matplot()
    ap.plot.weighted_pointmap(ax, points, bounding_box=bounding_box, color_gradient=[
                              "#0000FF"], opacity=1.0, coordinate_system="EPSG:4326")
    _finalize(ax, "test_weighted_0_0")

    ax = _get_matplot()
    ap.plot.weighted_pointmap(ax, points, color_weights=arr_c, bounding_box=bounding_box, color_gradient=[
                              "#0000FF", "#FF0000"], color_bound=[1, 5], opacity=1.0, coordinate_system="EPSG:4326")
    _finalize(ax, "test_weighted_1_0")

    ax = _get_matplot()
    ap.plot.weighted_pointmap(ax, points, size_weights=arr_s, bounding_box=bounding_box, color_gradient=[
                              "#0000FF"], size_bound=[1, 10], opacity=1.0, coordinate_system="EPSG:4326")
    _finalize(ax, "test_weighted_0_1")

    ax = _get_matplot()
    ap.plot.weighted_pointmap(ax, points, color_weights=arr_c, size_weights=arr_s, bounding_box=bounding_box, color_gradient=[
                              "#0000FF", "#FF0000"], color_bound=[1, 5], size_bound=[1, 10], opacity=1.0, coordinate_system="EPSG:4326")
    _finalize(ax, "test_weighted_1_1")


def test_heatmap():
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

    ax = _get_matplot()
    ap.plot.heatmap(ax, points, arr_c, bounding_box=bounding_box, map_zoom_level=13, coordinate_system='EPSG:4326')
    _finalize(ax, "test_heat_map1")


def test_choroplethmap():
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
    arr_wkb = arctern.ST_GeomFromText(arr_wkt)
    arr_count = pandas.Series(count_data)

    ax = _get_matplot()
    ap.plot.choroplethmap(ax, arr_wkb, arr_count, bounding_box=bounding_box, color_gradient=[
                          "#0000FF", "#FF0000"], color_bound=[2.5, 5], opacity=1.0, coordinate_system='EPSG:4326')
    _finalize(ax, "test_choropleth_map1")


def test_iconviz():
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

    ax = _get_matplot()
    ap.plot.iconviz(ax, points, bounding_box=bounding_box,
                    icon_path=png_path, coordinate_system="EPSG:4326")
    _finalize(ax, "test_icon_viz")

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

    ax = _get_matplot()
    ap.plot.fishnetmap(ax, points, arr_c, bounding_box=bounding_box, color_gradient=["#0000FF", "#FF0000"], cell_size=4, cell_spacing=1, opacity=1.0, coordinate_system='EPSG:4326')
    _finalize(ax, "test_fishnetmap")
