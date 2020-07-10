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

# pylint: disable=too-many-lines
# pylint: disable=too-many-public-methods, unused-argument, redefined-builtin

import numpy as np
import arctern
from arctern import GeoDataFrame, GeoSeries

def test_read_and_save_file_1():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["EPSG:4326", "EPSG:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1")
    read_gdf = arctern.read_file(filename="/tmp/test.shp")
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:4326"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"]


def test_read_and_save_file_2():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1")
    read_gdf = arctern.read_file(filename="/tmp/test.shp")
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:4326"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"]


def test_read_and_save_file_3():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = arctern.read_file(filename="/tmp/test.shp")
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:3857"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"]


def test_read_and_save_file_4():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = arctern.read_file(filename="/tmp/test.shp", bbox=(0, 0, 1, 1))
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:3857"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)"]


def test_read_and_save_file_5():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = arctern.read_file(filename="/tmp/test.shp", bbox=(0, 0, 1, 1))
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:3857"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)"]


def test_read_and_save_file_6():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    bbox = GeoSeries(["POLYGON ((0 0,1 0,1 1,0 1,0 0))"])
    read_gdf = arctern.read_file(filename="/tmp/test.shp", bbox=bbox)
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:3857"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)"]


def test_read_and_save_file_7():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = arctern.read_file(filename="/tmp/test.shp", rows=3)
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:3857"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)"]


def test_read_and_save_file_8():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    mask = GeoSeries(["POINT (3 3)", "POINT (4 4)"])
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = arctern.read_file(filename="/tmp/test.shp", mask=mask, rows=1)
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:3857"
    assert read_gdf["geo2"].values.tolist() == ["POINT (4 4)"]


def test_read_and_save_file_9():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["EPSG:4326", "EPSG:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1", index=True)
    read_gdf = arctern.read_file(filename="/tmp/test.shp")
    assert len(read_gdf.columns.values) == 7


def test_read_and_save_file_10():
    from collections import OrderedDict
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["EPSG:4326", "EPSG:3857"])
    arctern.to_file(gdf, filename="/tmp/test.shp", geometry="geo1",
                    schema={'geometry': 'Point', 'properties':
                    OrderedDict([('A', 'int'), ('B', 'float'),
                    ('other_geom', 'int'), ('geo2', 'str'), ('geo3', 'str')])})
    read_gdf = arctern.read_file(filename="/tmp/test.shp")
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:4326"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)",
                                                "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"]
