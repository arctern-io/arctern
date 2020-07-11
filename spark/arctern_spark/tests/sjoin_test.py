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

import pytest

import arctern_spark
from arctern_spark.geodataframe import GeoDataFrame

points = ["POINT (1 1)",
          "POINT (1 2)",
          "POINT (2 1)",
          "POINT (2 2)",
          "POINT (3 3)",
          "POINT (4 5)",
          "POINT (8 8)",
          "POINT (10 10)",
          ]

polygons = [
    "POLYGON ((0 0, 3 0, 3.1 3.1, 0 3, 0 0))",
    "POLYGON ((6 6, 3 6, 2.9 2.9, 6 3, 6 6))",
    "POLYGON ((6 6, 9 6, 9 9, 6 9, 6 6))",
    "POLYGON ((100 100, 100 101, 101 101, 101 100, 100 100))",
]


@pytest.fixture()
def left_df():
    yield GeoDataFrame({"A": range(8), "points": points}, geometries=["points"], crs="EPSG:4326")


@pytest.fixture()
def right_df():
    yield GeoDataFrame({"A": range(4), "polygons": polygons}, geometries=['polygons'], crs="EPSG:4326")


def test_right_within(left_df, right_df):
    r = arctern_spark.sjoin(left_df, right_df, "points", "polygons", "right")
    r.sort_values(by="A_left", inplace=True)
    assert r["points"].to_wkt().to_list() == [points[0], points[1], points[2], points[3], points[4], points[4],
                                              points[5], points[6], None]


def test_left_within(left_df, right_df):
    r = arctern_spark.sjoin(left_df, right_df, "points", "polygons", "left")
    r.sort_values(by="A_left", inplace=True)
    assert r["points"].to_wkt().to_list() == ['POINT (1 1)', 'POINT (1 2)', 'POINT (2 1)', 'POINT (2 2)',
                                              'POINT (3 3)', 'POINT (3 3)', 'POINT (4 5)', 'POINT (8 8)',
                                              'POINT (10 10)']


def test_full_within(left_df, right_df):
    r = arctern_spark.sjoin(left_df, right_df, "points", "polygons", "full")
    r.sort_values(by="A_left", inplace=True)
    assert r["points"].to_wkt().to_list() == ['POINT (1 1)', 'POINT (1 2)', 'POINT (2 1)', 'POINT (2 2)', 'POINT (3 3)',
                                              'POINT (3 3)', 'POINT (4 5)', 'POINT (8 8)', 'POINT (10 10)', None]


def test_inner_within(left_df, right_df):
    r = arctern_spark.sjoin(left_df, right_df, "points", "polygons", "inner")
    r.sort_values(by="A_left", inplace=True)
    assert r["points"].to_wkt().to_list() == ['POINT (1 1)', 'POINT (1 2)', 'POINT (2 1)', 'POINT (2 2)', 'POINT (3 3)',
                                              'POINT (3 3)', 'POINT (4 5)',
                                              'POINT (8 8)']


def test_inner_contains(left_df, right_df):
    r = arctern_spark.sjoin(right_df, left_df, "polygons", "points", how="inner", op="contains")
    r.sort_values(by="A_left", inplace=True)
    assert r["points"].to_wkt().to_list() == ['POINT (3 3)', 'POINT (1 1)', 'POINT (1 2)', 'POINT (2 2)', 'POINT (2 1)',
                                              'POINT (3 3)', 'POINT (4 5)', 'POINT (8 8)']
