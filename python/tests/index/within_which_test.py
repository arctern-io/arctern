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

import arctern
import pandas

def test_within_which():
    # assert False
    from arctern import GeoSeries, within_which
    left = GeoSeries(["Point(0 0)", "Point(1000 1000)", "Point(10 10)"])
    right = GeoSeries(["Polygon((9 10, 11 12, 11 8, 9 10))", "POLYGON ((-1 0, 1 2, 1 -2, -1 0))"])
    index = right.sindex
    res = index.within_which(left)
    print(res)
    assert len(res) == 3
    assert res[0] == 1
    assert res[1] is pandas.NA
    assert res[2] == 0

    left = GeoSeries(["Point(0 0)", "Point(1000 1000)", "Point(10 10)"], index=['A', 'B', 'C'])
    right = GeoSeries(["Polygon((9 10, 11 12, 11 8, 9 10))",
                       "Polygon((-1 0, 1 2, 1 -2, -1 0))"], index=['x', 'y'])
    index = right.sindex
    res = index.within_which(left)
    print(res)
    assert len(res) == 3
    assert res['A'] == 'y'
    assert res['B'] is pandas.NA
    assert res['C'] == 'x'
