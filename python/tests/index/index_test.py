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

def test_near_road():
    roads = arctern.GeoSeries(["LINESTRING (0 0,2 0)", "LINESTRING (5 0,5 5)"])
    points = arctern.GeoSeries(["POINT (1.0001 0.0001)"])
    index = roads.sindex
    result = index.near_road(points, 100)
    

