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

import arctern_spark
from arctern_spark import GeoSeries

point = ["POINT (-73.961003 40.760594)",
         "POINT (-73.959908 40.776353)",
         "POINT (-73.955183 40.773459)",
         "POINT (-73.985233 40.744682)",
         "POINT (-73.997969 40.682816)",
         "POINT (-73.996458 40.758197)",
         "POINT (-73.988240 40.748960)",
         "POINT (-73.985185 40.735828)",
         "POINT (-73.989726 40.767795)",
         "POINT (-73.992669 40.768327)"
         ]

road = ["LINESTRING (-73.9975944 40.7140611, -73.9974922 40.7139962)",
        "LINESTRING (-73.9980065 40.7138119, -73.9980743 40.7137811)",
        "LINESTRING (-73.9975554 40.7141073, -73.9975944 40.7140611)",
        "LINESTRING (-73.9978864 40.7143170, -73.9976740 40.7140968)",
        "LINESTRING (-73.9979810 40.7136728, -73.9980743 40.7137811)",
        "LINESTRING (-73.9980743 40.7137811, -73.9984728 40.7136003)",
        "LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)",
        "LINESTRING (-73.9594166 40.7593773, -73.9593736 40.7593593)",
        "LINESTRING (-73.9616090 40.7602969, -73.9615014 40.7602517)",
        "LINESTRING (-73.9615569 40.7601753, -73.9615014 40.7602517)"
        ]

points = GeoSeries(point, crs="EPSG:4326")
roads = GeoSeries(road, crs="EPSG:4326")


def test_near_road():
    r = arctern_spark.near_road(roads, points)
    assert r[0]
    assert not r[1:9].any()


def test_nearest_road():
    r = arctern_spark.nearest_road(roads, points).to_wkt()
    assert r[0] == "LINESTRING (-73.9611014 40.7608112, -73.9610636 40.7608639)"
    for i in range(1, 10):
        assert r[i] == "LINESTRING EMPTY"


def test_nearest_location_on_road():
    r = arctern_spark.nearest_location_on_road(roads, points).to_wkt()
    assert r[0] == "POINT (-73.9611014 40.7608112)"
    for i in range(1, 10):
        assert r[i] == "GEOMETRYCOLLECTION EMPTY"
