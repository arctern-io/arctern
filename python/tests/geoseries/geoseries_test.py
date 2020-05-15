from arctern.geoseries import GeoSeries
from arctern.geoseries.geoarray import GeoArray, GeoDtype
from arctern._wrapper_func import *
import pytest
import pandas._testing  as tm


def make_point(x, y):
    return "Point (%s %s)" % (x, y)


@pytest.fixture(params=['wkt', 'wkb'])
def sequence(request):
    if request.param == 'wkt':
        return [make_point(x, x) for x in range(5)]
    else:
        return ST_GeomFromText([make_point(x, x) for x in range(5)]).tolist()


@pytest.fixture(params=[1, 1.0, ('1', '1')])
def wrong_type_data(request):
    return [request.param for _ in range(5)]


@pytest.fixture(params=['wkt', 'wkb'])
def dic(request):
    if request.param == 'wkt':
        return {x: make_point(x, x) for x in range(5)}
    else:
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
        tm.assert_series_equal(s, expected_series, check_dtype=False)

    def test_from_dict(self, dic, expected_series):
        s = GeoSeries(dic)
        assert_is_geoseries(s)
        tm.assert_series_equal(s, expected_series, check_dtype=False)

    def test_from_empty(self):
        s = GeoSeries([])
        assert_is_geoseries(s)
        assert len(s) == 0

    def test_from_series(self, expected_series):
        s = GeoSeries(expected_series)
        assert_is_geoseries(s)
        tm.assert_series_equal(s, expected_series, check_dtype=False)

    def test_from_wrong_type_data(self, wrong_type_data):
        with pytest.raises(TypeError):
            GeoSeries(wrong_type_data)

    # only support None as na value
    def test_from_with_none_data(self):
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


class TestPandasMethod:
    def test_missing_values(self):
        s = GeoSeries(['Point (1 2)', None])
        assert s[1] is None
        assert s.isna().tolist() == [False, True]
        assert s.notna().tolist() == [True, False]
        assert not s.dropna().isna().any()


if __name__ == "__main__":
    from pandas import Series

    s = GeoSeries(['Point (1 2)', None])
    print(s.dropna().isna().all())
