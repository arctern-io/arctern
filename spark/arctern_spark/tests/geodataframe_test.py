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
import pandas as pd
import pytest
from databricks.koalas import Series, DataFrame

import arctern_spark
from arctern_spark.geodataframe import GeoDataFrame
from arctern_spark.geoseries import GeoSeries

wkt = "point (1 1)"
wkb = b'\x00\x00\x00\x00\x01?\xf0\x00\x00\x00\x00\x00\x00?\xf0\x00\x00\x00\x00\x00\x00'


class TestConstructor:
    @pytest.mark.parametrize("data", [
        wkt,
        wkb,
    ])
    def test_from_scalar_data(self, data):
        s = GeoDataFrame({'a': data}, index=[1], geometries=['a'])
        assert s['a'].to_wkt().to_list() == ["POINT (1 1)"]

    @pytest.mark.parametrize("data", [
        [wkt],
        [wkb],
        {0: wkt},
        {0: wkb},
        pd.Series(wkt),
        pd.Series(wkb),
    ])
    def test_from_pandas_data(self, data):
        s = GeoDataFrame({'a': data}, geometries=['a'])
        assert s['a'].to_wkt().to_list() == ["POINT (1 1)"]

    @pytest.mark.parametrize("data", [
        Series(wkt, name='a'),
        Series(wkb, name='a'),
        DataFrame({'a': [wkt]}),
        DataFrame({'a': [wkb]})
    ])
    def test_from_koalas(self, data):
        s = GeoDataFrame(data, geometries=['a'])
        assert s['a'].to_wkt().to_list() == ["POINT (1 1)"]


# test operation will not lose crs info
class TestCRS:
    def test_init_from_geoseries(self):
        crs = "EPSG:4326"
        gs = GeoSeries("point (1 2)", name='a', crs=crs)
        gdf = GeoDataFrame(gs)
        assert gdf._geometry_column_names == {'a'}
        assert gdf['a'].crs == crs

    def test_implicitly_set_geometries(self):
        crs = "EPSG:4326"
        psb = pd.Series("point (99 99)", name='b')
        psa = pd.Series("point (1 2)", name='a')
        gdf = GeoDataFrame({"a": psa, "b": psb}, geometries=['a'], crs=crs)
        assert gdf._geometry_column_names == {'a'}
        assert gdf['a'].crs == crs

    def test_explicitly_set_geometries(self):
        psb = pd.Series("point (99 99)", name='b')
        psa = pd.Series("point (1 2)", name='a')
        gdf = GeoDataFrame({"a": psa, "b": psb}, geometries=['a'], crs="EPSG:4326")
        assert gdf._geometry_column_names == {'a'}
        gdf.set_geometry('b', "EPSG:3857")
        assert gdf._geometry_column_names == {'a', 'b'}
        assert gdf['a'].crs == "EPSG:4326"
        assert gdf['b'].crs == "EPSG:3857"

    def test_setitem_getitem(self):
        # set or get item with scalar key
        gdf = GeoDataFrame([1], columns=['seq'])
        gdf['a'] = GeoSeries("point (1 2)", crs="EPSG:4326")
        gdf['b'] = GeoSeries("point (99 99)")
        assert gdf._geometry_column_names == {'a', 'b'}
        assert gdf['a'].crs == "EPSG:4326"
        assert gdf['b'].crs is None

        # set or get item with slice key
        gdf1 = GeoDataFrame([1], columns=['seq'])
        gdf1[['a', 'b']] = gdf[['a', 'b']]
        assert gdf1._geometry_column_names == {'a', 'b'}
        r = gdf1[:]
        assert r['a'].crs == "EPSG:4326"
        assert r['b'].crs is None

    def test_geoseries_modify_crs(self):
        gdf = GeoDataFrame(GeoSeries("point (1 2)", name='a', crs=None))
        assert gdf._geometry_column_names == {'a'}
        assert gdf['a'].crs is None

        # modify geoseries crs
        gdf['a'].crs = "EPSG:4326"
        assert gdf['a'].crs == "EPSG:4326"


class TestOp:
    def test_merge(self):
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
        assert isinstance(result, GeoDataFrame)
        assert isinstance(result["geometry"], GeoSeries)
        assert result._geometry_column_names == {'geometry', 'location'}
        assert result.location.crs == "EPSG:4326"

    def test_merge_same_column_name(self):
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
        assert isinstance(result, GeoDataFrame)
        assert isinstance(result["location_x"], GeoSeries)
        assert isinstance(result["location_y"], GeoSeries)
        assert result._geometry_column_names == {'location_x', 'location_y'}
        assert result.location_x.crs == "EPSG:4326"
        assert result.location_y.crs == "EPSG:3857"

    def test_merge_suffixed_column_name(self):
        data1 = {
            "A": range(5),
            "B": np.arange(5.0),
            "other_geom": range(5),
            "location_x": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        }
        gdf1 = GeoDataFrame(data1, geometries=["location_x"], crs=["epsg:4326"])
        data2 = {
            "A": range(5),
            "location_y": ["POINT (3 0)", "POINT (1 6)", "POINT (2 4)", "POINT (3 4)", "POINT (4 2)"],
        }
        gdf2 = GeoDataFrame(data2, geometries=["location_y"], crs=["epsg:3857"])
        result = gdf1.merge(gdf2, left_on="A", right_on="A")
        assert isinstance(result, GeoDataFrame)
        assert isinstance(result["location_x"], GeoSeries)
        assert result._geometry_column_names == {'location_x', 'location_y'}
        assert result.location_x.crs == "EPSG:4326"
        assert result.location_y.crs == "EPSG:3857"

    def test_disolve(self):
        data = {
            "A": range(5),
            "B": np.arange(5.0),
            "other_geom": [1, 1, 1, 2, 2],
            "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        }
        gdf = GeoDataFrame(data, geometries=["geo1"], crs=["epsg:4326"])
        dissolve_gdf = gdf.disolve(by="other_geom", col="geo1")
        assert dissolve_gdf["geo1"].to_wkt()[1] == "MULTIPOINT ((0 0), (1 1), (2 2))"
        assert dissolve_gdf["geo1"].to_wkt()[2] == "MULTIPOINT ((3 3), (4 4))"


def test_reset_index():
    data = {
        "A": range(5),
        "location": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
    }
    gdf = GeoDataFrame(data, index=[1, 2, 3, 4, 5], geometries=["location"], crs=["epsg:4326"])
    gdf.reset_index(inplace=True)
    assert isinstance(gdf, GeoDataFrame)
    assert gdf._geometry_column_names == {"location"}
    assert gdf._crs_for_cols["location"] == "epsg:4326"

    gdf1 = gdf.reset_index()
    assert isinstance(gdf1, GeoDataFrame)
    assert gdf1._geometry_column_names == {"location"}
    assert gdf1._crs_for_cols["location"] == "epsg:4326"


class TestFile:
    data = {
        "A": range(5),
        "B": np.arange(5.0),
        "other_geom": range(5),
        "geo1": ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)"],
        "geo2": ["POINT (1 1)", "POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)"],
        "geo3": ["POINT (2 2)", "POINT (3 3)", "POINT (4 4)", "POINT (5 5)", "POINT (6 6)"],
    }

    def test_read_and_save_file_1(self):
        gdf = GeoDataFrame(self.data, geometries=["geo1", "geo2"], crs=["EPSG:4326", "EPSG:3857"])
        gdf.to_file(filename="/tmp/test.shp", geometry="geo1")
        read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp")
        assert isinstance(read_gdf["geometry"], GeoSeries) is True
        assert read_gdf["geometry"].crs == "EPSG:4326"
        assert read_gdf["geo2"].to_pandas().sort_values('index').to_list() == ["POINT (1 1)", "POINT (2 2)",
                                                                               "POINT (3 3)", "POINT (4 4)",
                                                                               "POINT (5 5)"]

    def test_read_and_save_file_2(self):
        gdf = GeoDataFrame(self.data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
        arctern_spark.to_file(gdf, filename="/tmp/test.shp", geometry="geo1")
        read_gdf = arctern_spark.read_file(filename="/tmp/test.shp")
        assert isinstance(read_gdf["geometry"], GeoSeries) is True
        assert read_gdf["geometry"].crs == "EPSG:4326"
        assert read_gdf["geo2"].to_pandas().sort_values('index').tolist() == ["POINT (1 1)", "POINT (2 2)",
                                                                              "POINT (3 3)", "POINT (4 4)",
                                                                              "POINT (5 5)"]

    def test_read_and_save_file_3(self):
        gdf = GeoDataFrame(self.data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
        gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
        read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp")
        assert isinstance(read_gdf["geometry"], GeoSeries) is True
        assert read_gdf["geometry"].crs == "EPSG:3857"
        assert read_gdf["geo2"].to_pandas().sort_values('index').tolist() == ["POINT (1 1)", "POINT (2 2)",
                                                                              "POINT (3 3)", "POINT (4 4)",
                                                                              "POINT (5 5)"]

    def test_read_and_save_file_4(self):
        gdf = GeoDataFrame(self.data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
        gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
        read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp", bbox=(0, 0, 1, 1))
        assert isinstance(read_gdf["geometry"], GeoSeries) is True
        assert read_gdf["geometry"].crs == "EPSG:3857"
        assert read_gdf["geo2"].to_pandas().sort_values('index').tolist() == ["POINT (1 1)", "POINT (2 2)"]

    def test_read_and_save_file_5(self):
        gdf = GeoDataFrame(self.data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
        gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
        read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp", bbox=(0, 0, 1, 1))
        assert isinstance(read_gdf["geometry"], GeoSeries) is True
        assert read_gdf["geometry"].crs == "EPSG:3857"
        assert read_gdf["geo2"].to_pandas().sort_values('index').tolist() == ["POINT (1 1)", "POINT (2 2)"]

    def test_read_and_save_file_6(self):
        gdf = GeoDataFrame(self.data, geometries=["geo1", "geo2"], crs=["epsg:4326", "epsg:3857"])
        gdf.to_file(filename="/tmp/test.shp", geometry="geo1", crs="epsg:3857")
        bbox = GeoSeries(["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"])
        read_gdf = GeoDataFrame.from_file(filename="/tmp/test.shp", bbox=bbox)
        assert isinstance(read_gdf["geometry"], GeoSeries) is True
        assert read_gdf["geometry"].crs == "EPSG:3857"
        assert read_gdf["geo2"].to_pandas().sort_values('index').tolist() == ["POINT (1 1)", "POINT (2 2)"]


class TestJson:
    def test_to_json(self):
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

    def test_to_json_with_missing_value(self):
        data = {
            "A": np.nan,
            "B": np.arange(1.0),
            "other_geom": range(1),
            "geometry": ["POINT (0 0)"],
        }
        gdf = GeoDataFrame(data, geometries=["geometry"], crs=["epsg:4326"])
        json = gdf.to_json(geometry="geometry", na="drop")
        assert json == '{"type": "FeatureCollection", ' \
                       '"features": [{"id": "0", "type": "Feature", ' \
                       '"properties": {"B": 0.0, "other_geom": 0}, ' \
                       '"geometry": {"type": "Point", "coordinates": [0.0, 0.0]}}]}'

    def test_to_json_show_bbox(self):
        data = {
            "A": range(1),
            "B": np.arange(1.0),
            "other_geom": range(1),
            "geometry": ["LINESTRING (1 2,4 5,7 8)", ],
        }
        gdf = GeoDataFrame(data, geometries=["geometry"], crs=["epsg:4326"])
        json = gdf.to_json(geometry="geometry", na="drop", show_bbox=True)
        assert json == '{"type": "FeatureCollection", ' \
                       '"features": [{"id": "0", "type": "Feature", ' \
                       '"properties": {"A": 0, "B": 0.0, "other_geom": 0}, ' \
                       '"geometry": {"type": "LineString", "coordinates": ' \
                       '[[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]}, ' \
                       '"bbox": [1.0, 2.0, 7.0, 8.0]}], "bbox": [1.0, 2.0, 7.0, 8.0]}'
