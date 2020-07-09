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

# pylint: disable=protected-access
import numpy as np
import pandas
import arctern
from arctern import GeoDataFrame, GeoSeries


def test_from_geopandas():
    import geopandas
    from shapely.geometry import Point
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geometry": [Point(x, y) for x, y in zip(range(5), range(5))],
        "copy_geo": [Point(x + 1, y + 1) for x, y in zip(range(5), range(5))],
    }
    pdf = geopandas.GeoDataFrame(data, geometry="geometry", crs='epsg:4326')
    assert isinstance(pdf["geometry"], geopandas.GeoSeries)
    gdf = GeoDataFrame.from_geopandas(pdf)
    assert gdf.geometries_name == ["geometry", "copy_geo"]
    assert isinstance(gdf['geometry'], arctern.geoseries.GeoSeries) is True
    assert isinstance(gdf['copy_geo'], arctern.geoseries.GeoSeries) is True


def test_from_geoseries():
    data = GeoSeries(["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"], crs="epsg:4326")
    gdf = GeoDataFrame(data)
    assert isinstance(gdf[0], GeoSeries) is True
    assert gdf[0].crs == "EPSG:4326"


def test_to_geopandas():
    import geopandas
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
    pdf = gdf.to_geopandas()
    pdf.set_geometry("geo1", inplace=True)
    assert pdf.geometry.name == "geo1"
    assert isinstance(pdf["geo1"], geopandas.GeoSeries) is True


def test_dissolve_1():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": [1, 1, 1, 2, 2],
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
    dissolve_gdf = gdf.disolve(by="other_geom", col="geo1")
    assert dissolve_gdf["geo1"].to_wkt()[1] == "MULTIPOINT (0 0,1 1,2 2)"
    assert dissolve_gdf["geo1"].to_wkt()[2] == "MULTIPOINT (3 3,4 4)"


def test_dissolve_2():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": [1, 1, 1, 2, 2],
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
    dissolve_gdf = gdf.disolve(by="other_geom", col="geo1", aggfunc=np.mean)
    assert dissolve_gdf["geo1"].to_wkt()[1] == "MULTIPOINT (0 0,1 1,2 2)"
    assert dissolve_gdf["geo1"].to_wkt()[2] == "MULTIPOINT (3 3,4 4)"
    assert dissolve_gdf["A"][1] == 1
    assert dissolve_gdf["A"][2] == 3.5


def test_set_geometry():
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }
    gdf = GeoDataFrame(data, geometries=["geo1", "geo2"], crs=["epsg:4326", "EPSG:5626"])
    assert isinstance(gdf["geo1"], arctern.GeoSeries)
    assert isinstance(gdf["geo2"], pandas.Series)
    assert isinstance(gdf["geo3"], pandas.Series)
    gdf.set_geometry(col="geo1", crs="epsg:3857", inplace=True)
    gdf.set_geometry(col="geo3", inplace=True)
    assert gdf["geo1"].crs == "EPSG:3857"
    assert gdf["geo2"].crs == "EPSG:5626"
    assert gdf["geo3"].crs is None
    assert gdf._geometry_column_name == ["geo1", "geo2", "geo3"]


def test_merge1():
    data1 = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geometry": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
    }
    gdf1 = GeoDataFrame(data1, geometries=["geometry"], crs=["epsg:4326"])
    data2 = {
        "A": range(5),
        "location": ["POINT (3 0)", "POINT (1 6)", "POINT (2 4)", "POINT (3 4)", "POINT (4 2)"],
    }
    gdf2 = GeoDataFrame(data2, geometries=["location"], crs=["epsg:4326"])
    result = gdf1.merge(gdf2, left_on="A", right_on="A")
    assert isinstance(result, arctern.GeoDataFrame) is True
    assert isinstance(result["geometry"], arctern.GeoSeries) is True
    assert result.geometry.crs == "EPSG:4326"


def test_merge2():
    data1 = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "location": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
    }
    gdf1 = GeoDataFrame(data1, geometries=["location"], crs=["epsg:4326"])
    data2 = {
        "A": range(5),
        "location": ["POINT (3 0)", "POINT (1 6)", "POINT (2 4)", "POINT (3 4)", "POINT (4 2)"],
    }
    gdf2 = GeoDataFrame(data2, geometries=["location"], crs=["epsg:3857"])
    result = gdf1.merge(gdf2, left_on="A", right_on="A")
    assert isinstance(result, arctern.GeoDataFrame) is True
    assert isinstance(result["location_x"], arctern.GeoSeries) is True
    assert result.location_x.crs == "EPSG:4326"


def test_to_json_1():
    data = {
        "A": range(1),
        "B": np.arange(1.0),
        "other_geom": range(1),
        "geometry": ["POINT (0 0)"],
    }
    gdf = GeoDataFrame(data, geometries=["geometry"], crs=["epsg:4326"])
    json = gdf.to_json(geometry="geometry")
    assert json == '{"type": "FeatureCollection", ' \
                   '"features": [{"id": "0", "type": "Feature", ' \
                   '"properties": {"A": 0, "B": 0.0, "other_geom": 0}, ' \
                   '"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]}'


def test_to_json_2():
    data = {
        "A" : np.nan,
        "B" : np.arange(1.0),
        "other_geom": range(1),
        "geometry": ["POINT (0 0)"],
    }
    gdf = GeoDataFrame(data, geometries=["geometry"], crs=["epsg:4326"])
    json = gdf.to_json(geometry="geometry", na="drop")
    assert json == '{"type": "FeatureCollection", ' \
                   '"features": [{"id": "0", "type": "Feature", ' \
                   '"properties": {"B": 0.0, "other_geom": 0}, ' \
                   '"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]}'


def test_to_json_3():
    data = {
        "A" : range(1),
        "B" : np.arange(1.0),
        "other_geom": range(1),
        "geometry": ["LINESTRING (1 2,4 5,7 8)",],
    }
    gdf = GeoDataFrame(data, geometries=["geometry"], crs=["epsg:4326"])
    json = gdf.to_json(geometry="geometry", na="drop", show_bbox=True)
    assert json == '{"type": "FeatureCollection", ' \
                   '"features": [{"id": "0", "type": "Feature", ' \
                   '"properties": {"A": 0, "B": 0.0, "other_geom": 0}, ' \
                   '"geometry": {"type": "LineString", "coordinates": ' \
                   '[[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]}, ' \
                   '"bbox": [1.0, 2.0, 7.0, 8.0]}], "bbox": [1.0, 2.0, 7.0, 8.0]}'


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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1")
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp")
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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp")
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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp", bbox=(0, 0, 1, 1))
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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp", bbox=(0, 0, 1, 1))
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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    bbox = GeoSeries(["POLYGON ((0 0,1 0,1 1,0 1,0 0))"])
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp", bbox=bbox)
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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp", rows=3)
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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp", mask=mask, rows=1)
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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1", index=True)
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp")
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
    gdf.to_file(filename="/tmp/test.shp", geometry="geo1",
                schema={'geometry': 'Point', 'properties':
                OrderedDict([('A', 'int'), ('B', 'float'),
                ('other_geom', 'int'), ('geo2', 'str'),
                ('geo3', 'str')])})
    read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp")
    assert isinstance(read_gdf["geometry"], GeoSeries) is True
    assert read_gdf["geometry"].crs == "EPSG:4326"
    assert read_gdf["geo2"].values.tolist() == ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"]
