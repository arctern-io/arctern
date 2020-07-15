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


bounding_box = [-73.998427, 40.730309, -73.954348, 40.780816]


def test_unique_value_choroplethmap():
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
    unique_value_infos = {1: 0xFF0000, 2: 0x00FF00, 3: 0x0000FF, 4: 0x00FFFF, 5: 0xFFFFFF}
    ap.plot.unique_value_choroplethmap(ax, arr_wkb, arr_count, bounding_box=bounding_box,
                                       unique_value_infos=unique_value_infos, opacity=1.0,
                                       coordinate_system='EPSG:4326')
    _finalize(ax, "test_unique_value_choropleth_map1")

