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

import pytest
from pandas.testing import assert_series_equal
from arctern.geoseries import GeoSeries
from arctern.geoseries.geoarray import GeoDtype
from arctern._wrapper_func import ST_GeomFromText


def make_point(x, y):
    return "POINT (%s %s)" % (x, y)


@pytest.fixture(params=['wkt', 'wkb'])
def sequence(request):
    if request.param == 'wkt':
        return [make_point(x, x) for x in range(5)]
    return ST_GeomFromText([make_point(x, x) for x in range(5)]).tolist()


@pytest.fixture(params=[1, 1.0, ('1', '1')])
def wrong_type_data(request):
    return [request.param for _ in range(5)]


@pytest.fixture(params=['wkt', 'wkb'])
def dic(request):
    if request.param == 'wkt':
        return {x: make_point(x, x) for x in range(5)}
    return {x: ST_GeomFromText(make_point(x, x))[0] for x in range(5)}


@pytest.fixture
def expected_series():
    return ST_GeomFromText([make_point(x, x) for x in range(5)])


def assert_is_geoseries(s):
    assert isinstance(s, GeoSeries)
    assert isinstance(s.dtype, GeoDtype)


class TestConstructor:

    def test_from_sequence(self, sequence, expected_series):
        s = GeoSeries(sequence)
        assert_is_geoseries(s)
        assert_series_equal(s, expected_series, check_dtype=False)

    def test_from_dict(self, dic, expected_series):
        s = GeoSeries(dic)
        assert_is_geoseries(s)
        assert_series_equal(s, expected_series, check_dtype=False)

    def test_from_empty(self):
        s = GeoSeries([], dtype="GeoDtype")
        assert_is_geoseries(s)
        assert len(s) == 0

    def test_from_series(self, expected_series):
        s = GeoSeries(expected_series)
        assert_is_geoseries(s)
        assert_series_equal(s, expected_series, check_dtype=False)

    def test_fom_geoarray(self, expected_series):
        s = GeoSeries(expected_series.values)
        assert_is_geoseries(s)
        assert_series_equal(s, expected_series, check_dtype=False)

    def test_from_wrong_type_data(self, wrong_type_data):
        with pytest.raises(TypeError):
            GeoSeries(wrong_type_data)

    # only support None as na value
    def test_from_with_na_data(self):
        s = GeoSeries(['Point (1 2)', None])
        assert_is_geoseries(s)
        assert len(s) == 2
        assert s.hasnans

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


class TestType:
    def setup_method(self):
        self.s = GeoSeries([make_point(x, x) for x in range(5)])

    def test_head_tail(self):
        assert_is_geoseries(self.s.head())
        assert_is_geoseries(self.s.tail())

    def test_slice(self):
        assert_is_geoseries(self.s[2::5])
        assert_is_geoseries(self.s[::-1])

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


# other method will be tested in geoarray_test.py
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

        s1 = s[::-1].fillna(method="bfill")
        assert s1[0] == s1[1]

        # set item with na value
        import numpy as np
        s[0] = np.nan
        assert s[0] is None

        import pandas as pd
        s[0] = pd.NA
        assert s[0] is None

    def test_equals(self):
        s1 = GeoSeries([make_point(1, 1), None])
        s2 = GeoSeries([make_point(1, 1), None])
        assert s1.equals(s2)

    def test_unique(self):
        s = GeoSeries([make_point(1, 1), make_point(1, 1), make_point(1, 2), None])
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
        assert s.astype('string').tolist() == [make_point(1, 1), make_point(1, 2)]



def test_geo_method_with_missing_value():
    s1 = GeoSeries([make_point(1, 1), None])
    s2 = GeoSeries([None, make_point(1, 1)])
    s3 = GeoSeries([make_point(1, 1), None])

    assert s1.geom_equals(s3).all()
    assert not s1.geom_equals(s2).any()

def test_geoseries_type_by_df_box_col_values():
    from pandas import DataFrame, Series
    series = GeoSeries(["POINT (0 0)", None, "POINT (0 1)", "POINT (2 0)"])
    df = DataFrame({'s':series})
    assert isinstance(df['s'], type(series))

    series = Series([1, None, 2, 3])
    df = DataFrame({'s':series})
    assert isinstance(df['s'], type(series))
