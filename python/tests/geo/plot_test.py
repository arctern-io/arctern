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

# pylint: disable=wrong-import-order

import os
import pandas
import arctern
import matplotlib.pyplot as plt

def test_plot1():
    raw_data = []
    raw_data.append('polygon((0 0,0 1,1 1,1 0,0 0))')
    raw_data.append('linestring(0 0,0 1,1 1,1 0,0 0)')
    raw_data.append('point(2 2)')
    raw_data.append("GEOMETRYCOLLECTION(" \
                    "MULTIPOLYGON (((0 0,0 1,1 1,1 0,0 0)),((1 1,1 2,2 2,2 1,1 1)))," \
                    "POLYGON((3 3,3 4,4 4,4 3,3 3))," \
                    "LINESTRING(0 8,5 5,8 0)," \
                    "POINT(4 7)," \
                    "MULTILINESTRING ((1 1,1 2),(2 4,1 9,1 8))," \
                    "MULTIPOINT (6 8,5 7)" \
                    ")")
    arr_wkt = pandas.Series(raw_data)
    arr_wkb = arctern.ST_GeomFromText(arr_wkt)

    file_name = "/tmp/test_plot1.png"

    if os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        assert False

    fig, ax = plt.subplots()
    arctern.plot(ax, arr_wkb)
    ax.grid()
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    assert 15 <= file_size <= 25

def test_plot2():
    raw_data = []
    raw_data.append('point(0 0)')
    raw_data.append('point(1 1)')
    raw_data.append('point(2 2)')

    arr_wkt = pandas.Series(raw_data)
    arr_wkb = arctern.ST_GeomFromText(arr_wkt)

    file_name = "/tmp/test_plot2.png"

    if os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        assert False

    fig, ax = plt.subplots()
    arctern.plot(ax, arr_wkb, color=['red', 'blue', 'black'], marker='^', markersize=100)
    ax.grid()
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    assert 10 <= file_size <= 15

def test_plot3():
    raw_data = []
    raw_data.append('linestring(0 0, 5 5, 10 10)')
    raw_data.append('linestring(0 10, 5 5, 10 0)')
    raw_data.append('linestring(0 1, 5 6, 10 11)')
    raw_data.append('linestring(0 11, 5 6, 10 1)')

    arr_wkt = pandas.Series(raw_data)
    arr_wkb = arctern.ST_GeomFromText(arr_wkt)

    file_name = "/tmp/test_plot3.png"

    if os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        assert False

    fig, ax = plt.subplots()
    arctern.plot(ax, arr_wkb,
                      color=['green', 'red', 'black', 'orange'],
                      linewidth=[5, 6, 7, 8],
                      linestyle=['solid', 'dashed', 'dashdot', 'dotted'])
    ax.grid()
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    assert 30 <= file_size <= 50

def test_plot4():
    raw_data = []
    raw_data.append('polygon((0 0,0 1,1 1,1 0,0 0))')
    raw_data.append('polygon((1 1,1 2,2 2,2 1,1 1))')
    raw_data.append('polygon((2 2,2 3,3 3,3 2,2 2))')
    raw_data.append('polygon((3 3,3 4,4 4,4 3,3 3))')

    arr_wkt = pandas.Series(raw_data)
    arr_wkb = arctern.ST_GeomFromText(arr_wkt)

    file_name = "/tmp/test_plot4.png"

    if os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        assert False

    fig, ax = plt.subplots()
    arctern.plot(ax, arr_wkb,
                      edgecolor=['green', 'red', 'black', 'orange'],
                      linewidth=[5, 6, 7, 8],
                      linestyle=['solid', 'dashed', 'dashdot', 'dotted'],
                      facecolor=['red', 'black', 'orange', 'green'])
    ax.grid()
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    assert 10 <= file_size <= 20

def test_plot5():
    raw_data = []
    raw_data.append('circularstring(-2 -2, 2 2, -2 -2)')
    raw_data.append('circularstring(-1 -1, 1 1, -1 -1)')

    arr_wkt = pandas.Series(raw_data)
    arr_wkb = arctern.ST_CurveToLine(arctern.ST_GeomFromText(arr_wkt))

    file_name = "/tmp/test_plot5.png"

    if os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        assert False

    fig, ax = plt.subplots()
    arctern.plot(ax, arr_wkb)
    ax.grid()
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    assert 25 <= file_size <= 35

def test_plot6():
    raw_data = []
    raw_data.append('point(0 0)')
    raw_data.append('linestring(0 10, 5 5, 10 0)')
    raw_data.append('polygon((2 2,2 3,3 3,3 2,2 2))')
    raw_data.append("GEOMETRYCOLLECTION(" \
                    "polygon((1 1,1 2,2 2,2 1,1 1))," \
                    "linestring(0 1, 5 6, 10 11)," \
                    "POINT(4 7))")

    arr_wkt = pandas.Series(raw_data)
    arr_wkb = arctern.ST_CurveToLine(arctern.ST_GeomFromText(arr_wkt))

    file_name = "/tmp/test_plot6.png"

    if os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        assert False

    fig, ax = plt.subplots()
    arctern.plot(ax, arr_wkb,
                      color=['orange', 'green'],
                      marker='^',
                      markersize=100,
                      linewidth=[None, 7, 8],
                      linestyle=[None, 'dashed', 'dashdot'],
                      edgecolor=[None, None, 'red'],
                      facecolor=[None, None, 'black'])
    ax.grid()
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    # print(file_size)
    assert 20 <= file_size <= 30

def test_plot7():
    raw_data = []
    raw_data.append('point(0 0)')
    raw_data.append('linestring(0 10, 5 5, 10 0)')
    raw_data.append('polygon((2 2,2 3,3 3,3 2,2 2))')
    raw_data.append("GEOMETRYCOLLECTION(" \
                    "polygon((1 1,1 2,2 2,2 1,1 1))," \
                    "linestring(0 1, 5 6, 10 11)," \
                    "POINT(4 7))")

    arr_wkt = pandas.Series(raw_data)
    arr_wkb = arctern.ST_CurveToLine(arctern.ST_GeomFromText(arr_wkt))
    df = pandas.DataFrame({'wkb':arr_wkb})

    file_name = "/tmp/test_plot7.png"

    if os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        assert False

    fig, ax = plt.subplots()
    arctern.plot(ax, df,
                      color=['orange', 'green'],
                      marker='^',
                      markersize=[100],
                      linewidth=[None, 7, 8],
                      linestyle=[None, 'dashed', 'dashdot'],
                      edgecolor=[None, None, 'red'],
                      facecolor=[None, None, 'black'])
    ax.grid()
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    # print(file_size)
    assert 20 <= file_size <= 30

def test_plot8():
    raw_data = []
    raw_data.append('point(0 0)')
    raw_data.append('linestring(0 10, 5 5, 10 0)')
    raw_data.append('polygon((2 2,2 3,3 3,3 2,2 2))')
    raw_data.append("GEOMETRYCOLLECTION(" \
                    "polygon((1 1,1 2,2 2,2 1,1 1))," \
                    "linestring(0 1, 5 6, 10 11)," \
                    "POINT(4 7))")

    arr_wkt = pandas.Series(raw_data)
    arr_wkb = arctern.ST_CurveToLine(arctern.ST_GeomFromText(arr_wkt))

    file_name = "/tmp/test_plot8.png"

    if os.path.exists(file_name):
        os.remove(file_name)

    if os.path.exists(file_name):
        assert False

    fig, ax = plt.subplots()
    arctern.plot(ax, arr_wkb,
                      color=['orange', 'green'],
                      marker='^',
                      markersize=100,
                      alpha=0.6,
                      linewidth=[None, 7, 8],
                      linestyle=[None, 'dashed', 'dashdot'],
                      edgecolor=[None, None, 'red'],
                      facecolor=[None, None, 'black'])
    ax.grid()
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    # print(file_size)
    assert 20 <= file_size <= 30
