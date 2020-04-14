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
import os
import pandas
import matplotlib.pyplot as plt
import arctern

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
    fig.savefig(file_name)
    file_size = os.path.getsize(file_name)
    file_size = file_size / 1024
    assert 15 <= file_size <= 25
