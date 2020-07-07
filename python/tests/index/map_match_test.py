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
    rst = arctern.near_road(roads, points, 100)
    assert len(rst) == 1
    assert rst[0]

def test_nearest_location_on_road():
    roads = arctern.GeoSeries("LINESTRING (1 2,1 3)")
    gps_points = arctern.GeoSeries("POINT (1.0001 2.5)")
    rst = arctern.ST_AsText(arctern.nearest_location_on_road(roads, gps_points))
    assert len(rst) == 1
    assert rst[0] == "POINT (1.0 2.5)"

def test_nearest_road():
    roads = arctern.GeoSeries(["LINESTRING (0 0,2 0)", "LINESTRING (5 0,5 5)"])
    gps_points = arctern.GeoSeries("POINT (1.0001 1)")
    rst = arctern.ST_AsText(arctern.nearest_road(roads, gps_points))
    assert len(rst) == 1
    assert rst[0] == "LINESTRING (0 0,2 0)"


def query():
    roads = arctern.GeoSeries(["LINESTRING (-73.9975944 40.7140611,-73.9974922 40.7139962)",
                               "LINESTRING (-73.9980065 40.7138119,-73.9980743 40.7137811)"])
    gps_points = arctern.GeoSeries("LINESTRING (-73.9975944 40.7140611,-73.9974922 40.7139962)")
    index_tree = roads.sindex
    rst = arctern.ST_AsText(index_tree.query(gps_points))
    assert len(rst) == 1
    assert rst[0] == "LINESTRING (-73.9975944 40.7140611,-73.9974922 40.7139962)"
