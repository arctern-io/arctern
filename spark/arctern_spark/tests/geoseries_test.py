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

# pylint: disable=attribute-defined-outside-init, redefined-outer-name
import os
from osgeo import ogr
import numpy as np
import pandas as pd
import pytest
import databricks.koalas as ks
from arctern_spark import GeoSeries


def make_point(x, y):
    return "POINT (%s %s)" % (x, y)


@pytest.fixture(params=['wkt', 'wkb'])
def sequence(request):
    if request.param == 'wkt':
        return [make_point(x, x) for x in range(5)]
    return [ogr.CreateGeometryFromWkt(make_point(x, x)).ExportToWkb() for x in range(5)]


@pytest.fixture(params=[1, 1.0])
def wrong_type_data(request):
    return [request.param for _ in range(5)]


@pytest.fixture(params=['wkt', 'wkb'])
def dic(request):
    if request.param == 'wkt':
        return {x: make_point(x, x) for x in range(5)}
    return {x: ogr.CreateGeometryFromWkt(make_point(x, x)).ExportToWkb() for x in range(5)}


@pytest.fixture
def expected_series():
    return [ogr.CreateGeometryFromWkt(make_point(x, x)).ExportToWkb() for x in range(5)]


def assert_is_geoseries(s):
    assert isinstance(s, GeoSeries)
    assert isinstance(s.dtype, object)


class TestConstructor:

    def test_from_sequence(self, sequence, expected_series):
        s = GeoSeries(sequence)
        assert_is_geoseries(s)
        assert s.to_list() == expected_series

    def test_from_dict(self, dic, expected_series):
        s = GeoSeries(dic)
        assert_is_geoseries(s)
        assert s.to_list() == expected_series

    def test_from_empty(self):
        s = GeoSeries([])
        assert_is_geoseries(s)
        assert len(s) == 0

        s = GeoSeries()
        assert_is_geoseries(s)
        assert len(s) == 0

    def test_explicate_dtype(self, sequence, expected_series):
        s = GeoSeries(sequence)
        assert_is_geoseries(s)
        assert s.to_list() == expected_series

    def test_from_series(self, expected_series):
        s = GeoSeries(expected_series)
        assert_is_geoseries(s)
        assert s.to_list() == expected_series

    def test_from_wrong_type_data(self, wrong_type_data):
        with pytest.raises(TypeError):
            GeoSeries(wrong_type_data)

    def test_from_with_na_data(self):
        s = GeoSeries(['Point (1 2)', None, np.nan])
        assert_is_geoseries(s)
        assert len(s) == 3
        assert s.hasnans
        assert s[1] is None
        assert s[2] is None

    def test_from_mismatch_crs(self):
        data = GeoSeries('Point (1 2)', crs='epsg:7777')
        with pytest.raises(ValueError):
            GeoSeries(data, crs='epsg:4326')

    def test_form_scalar_with_index(self):
        index = [1, 2, 3, 4, 5]
        s = GeoSeries('Point (1 2)', index=index)
        assert_is_geoseries(s)
        assert len(s) == 5
        for i in index:
            assert s[i] == s[index[0]]

    def test_form_series_with_index(self):
        index = [1, 2, 3, 4, 5]
        s = pd.Series(make_point(1, 1), index=index)
        geo_s = GeoSeries(s)
        assert_is_geoseries(geo_s)
        assert len(geo_s) == 5
        for i in index:
            assert geo_s[i] == geo_s[index[0]]


class TestType:
    def setup_method(self):
        self.s = GeoSeries([make_point(x, x) for x in range(5)])

    def test_head_tail(self):
        assert_is_geoseries(self.s.head())
        # assert_is_geoseries(self.s.tail())

    def test_slice(self):
        assert_is_geoseries(self.s[2::5])
        assert_is_geoseries(self.s[1::-1])

    def test_loc_iloc(self):
        assert_is_geoseries(self.s.loc[1:])
        assert_is_geoseries(self.s.iloc[:4])

    def test_take(self):
        assert_is_geoseries(self.s.take([0, 2, 4]))

    def test_geom_method(self):
        assert_is_geoseries(self.s.buffer(0.2))
        assert_is_geoseries(self.s.intersection(self.s))
        assert_is_geoseries(self.s.centroid)


class TestCRS:
    def setup_method(self):
        self.crs = "epsg:3854"
        self.s = GeoSeries([make_point(x, x) for x in range(5)], crs=self.crs)

    def test_constrctor(self):
        assert self.crs.upper() == self.s.crs

    def test_series_method(self):
        assert self.s.head().crs == self.crs.upper()
        assert self.s[:4].crs == self.crs.upper()
        assert self.s.take([1, 3, 4]).crs == self.crs.upper()
        assert self.s[[True, False, True, True, True]].crs == self.crs.upper()

    # test methods in GeoSeries will produce GeoSeries as result
    def test_geom_method(self):
        assert self.s.buffer(0.2).crs == self.crs.upper()
        assert self.s.intersection(self.s).crs == self.crs.upper()
        assert self.s.centroid.crs == self.crs.upper()


# other method will be tested in geoarray_test.py and series_method_test.py
class TestPandasMethod:
    def test_missing_values(self):
        s = GeoSeries([make_point(1, 2), None])
        assert s[1] is None
        assert s.isna().tolist() == [False, True]
        assert s.notna().tolist() == [True, False]
        assert not s.dropna().isna().any()

        s1 = s.fillna(make_point(1, 1))
        s1 = s1.to_wkt()
        assert s1[0] == "POINT (1 2)"
        assert s1[1] == "POINT (1 1)"

        # fillna with method
        s1 = s.fillna(method='ffill')
        assert s1[0] == s1[1]

    def test_equals(self):
        s1 = GeoSeries([make_point(1, 1), make_point(2, 2)])
        s2 = GeoSeries([make_point(1, 1), make_point(2, 2)])
        assert (s1.equals(s2)).all()

        s1 = GeoSeries()
        s2 = GeoSeries()
        assert (s1.equals(s2)).all()

    def test_unique(self):
        s = GeoSeries([make_point(1, 1), make_point(
            1, 1), make_point(1, 2), None])
        assert len(s.unique()) == 3

    def test_operator(self):
        # __eq__
        s = GeoSeries([make_point(0, 0), make_point(1, 2), None])
        r = s == s[0]
        assert r.tolist() == [True, False, False]

        # __ne__
        r = s != s[0]
        assert r.tolist() == [False, True, True]

    def test_astype(self):
        s = GeoSeries([make_point(1, 1), make_point(1, 2)])
        assert s.astype(str).tolist() == [make_point(1, 1), make_point(1, 2)]
        assert s.astype('string').tolist() == [
            make_point(1, 1), make_point(1, 2)]


class TestGeoMethods:
    def test_geom_equals_with_missing_value(self):
        s1 = GeoSeries([make_point(1, 1), None])
        s2 = GeoSeries([None, make_point(1, 1)])
        s3 = GeoSeries([make_point(1, 1), None])

        rst1 = s1.geom_equals(s3)
        assert rst1[0]
        assert not rst1[1]

        assert not s1.geom_equals(s2).any()

    def test_geom_with_index(self):
        index = [1, 2]

        # property
        s = GeoSeries([make_point(1, 1), None], index=index)
        s1 = s.length
        assert s1.index.to_pandas().to_list() == index
        assert s1[index[0]] == 0
        assert pd.isna(s1[index[1]])

        # unary
        s1 = s.precision_reduce(3)
        assert not pd.isna(s1[index[0]])
        assert pd.isna(s1[index[1]])

        # binary with same index
        left = GeoSeries([make_point(1, 1), None], index=index)
        right = GeoSeries([make_point(1, 1), None], index=index)
        s1 = left.geom_equals(right)
        assert s1.index.to_pandas().to_list() == index

        # binary with different index will align index
        left = GeoSeries([make_point(1, 1), None], index=[1, 2])
        right = GeoSeries([make_point(1, 1), None], index=[3, 4])
        s1 = left.geom_equals(right)
        assert s1.to_list() == [False, False, False, False]

    def test_to_wkb(self):
        s = GeoSeries(make_point(1, 1))
        s1 = s.to_wkb()
        assert isinstance(s1, ks.Series)


geometry_list = ["POINT (1 1)",
                 "MULTIPOINT (1 1,3 4)",
                 "POLYGON ((1 1,1 2,2 2,2 1,1 1))",
                 "MULTIPOLYGON (((1 1,1 2,2 2,2 1,1 1)),((0 0,-2 3,1 1,1 -1,0 0)))",
                 "LINESTRING (1 1,1 2,2 3,1 1)",
                 "MULTILINESTRING ((1 1,1 2),(2 4,1 9,1 8))",
                 # "GEOMETRYCOLLECTION ( LINESTRING ( 90 190, 120 190, 50 60, 130 10, 190 50, 160 90, 10 150, 90 190 ), POINT(90 190) )"
                 ]


@pytest.fixture(params=geometry_list)
def source(request):
    return GeoSeries([request.param] * 1, name="geometry", crs="EPSG:4326")


class TestFile:
    @pytest.mark.parametrize("driver, extension", [("ESRI Shapefile", "shp"), ("GeoJSON", "geojson")])
    def test_to_from_file(self, driver, extension, tmpdir, source):
        file_name = os.path.join(str(tmpdir), "test." + extension)
        source.to_file(file_name, driver=driver)

        dest = GeoSeries.from_file(file_name)
        assert (dest == source).all()
        source = source.to_crs("EPSG:3857")
        dest = dest.to_crs("EPSG:3857")
        assert (dest == source).all()

    @pytest.mark.parametrize("driver, extension", [("ESRI Shapefile", "shp"), ("GeoJSON", "geojson")])
    def test_complex_geometry(self, driver, extension, tmpdir):
        source = GeoSeries(geometry_list, name="geometry", crs="EPSG:4326")
        file_name = os.path.join(str(tmpdir), "test." + extension)
        if driver == "ESRI Shapefile":
            with pytest.xfail("ESRI shapefiles can only store one kind of geometry per layer"):
                source.to_file(file_name, driver=driver)
        else:
            source.to_file(file_name, driver=driver)
            dest = GeoSeries.from_file(file_name)
            assert (dest == source).all()
            source = source.to_crs("EPSG:3857")
            dest = dest.to_crs("EPSG:3857")
            assert (dest == source).all()

    @pytest.mark.parametrize("driver, extension", [("ESRI Shapefile", "shp"), ("GeoJSON", "geojson")])
    def test_empty_series(self, driver, extension, tmpdir):
        file_name = os.path.join(str(tmpdir), "test." + extension)
        source = GeoSeries([], crs="EPSG:4316", name="geometry")
        source.to_file(file_name, driver=driver)

        dest = GeoSeries.from_file(file_name)
        assert (dest == source).all()

    @pytest.mark.parametrize("driver, extension", [("ESRI Shapefile", "shp"), ("GeoJSON", "geojson")])
    def test_missing_geometry(self, driver, extension, tmpdir):
        file_name = os.path.join(str(tmpdir), "test." + extension)
        #source = GeoSeries([make_point(1, 1), None], name="geometry")
        source = GeoSeries([make_point(1, 1)], name="geometry")
        source.to_file(file_name, driver=driver)

        dest = GeoSeries.from_file(file_name)
        assert (dest == source).all()

    def test_from_esri_zip(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../python/tests/geoseries/dataset"))
        file_name = "zip://" + os.path.join(data_dir, "taxi_zones.zip")
        dest = GeoSeries.from_file(file_name)
        assert len(dest) == 9
        assert dest.crs == "EPSG:2263"
