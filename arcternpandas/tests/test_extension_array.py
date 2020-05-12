"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite (by inheriting the pandas test suite), and should
contain no other tests.
"""

import pytest
from pandas.tests.extension import base as extension_tests
from arcternpandas.geoarray import GeoArray, GeoDtype, from_wkt
import pandas as pd


# -----------------------------------------------------------------------------
# Required fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return GeoDtype()


def make_point(x, y):
    return "Point (%s %s)" % (x, y)


@pytest.fixture
def data():
    """Length-100 array for this type.

   * data[0] and data[1] should both be non missing
   * data[0] and data[1] should not be equal
   """
    wkt = [make_point(x, x) for x in range(100)]
    ga = from_wkt(wkt)
    return ga


@pytest.fixture
def data_for_twos():
    """Length-100 array in which all the elements are two."""
    raise NotImplementedError


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    ga = from_wkt(["invalid geo", make_point(1, 1)])
    return ga


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.

    Parameters
    ----------
    data : fixture implementing `data`

    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """

    def gen(count):
        for _ in range(count):
            yield data

    return gen


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    raise NotImplementedError


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    raise NotImplementedError


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.
    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.
    By default, uses ``operator.or``
    """
    return lambda x, y: x is None and y is None


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return None


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    return from_wkt([make_point(1, 1),
                     make_point(1, 1),
                     None,
                     None,
                     make_point(0, 0),
                     make_point(0, 0),
                     make_point(1, 1),
                     make_point(2, 2)])


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


@pytest.fixture(
    params=[
        lambda x: 1,
        lambda x: [1] * len(x),
        lambda x: pd.Series([1] * len(x)),
        lambda x: x,
    ],
    ids=["scalar", "list", "series", "object"],
)
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_frame(request):
    """
    Boolean fixture to support Series and Series.to_frame() comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    """
    Boolean fixture to support arr and Series(arr) comparison testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    """
    Boolean fixture to support comparison testing of ExtensionDtype array
    and numpy array.
    """
    return request.param


@pytest.fixture(params=["ffill", "bfill"])
def fillna_method(request):
    """
    Parametrized fixture giving method parameters 'ffill' and 'bfill' for
    Series.fillna(method=<method>) testing.
    """
    return request.param


@pytest.fixture(params=[True, False])
def as_array(request):
    """
    Boolean fixture to support ExtensionDtype _from_sequence method testing.
    """
    return request.param


# Fixtures defined in pandas/conftest.py that are also needed: defining them
# here instead of importing for compatibility


@pytest.fixture(
    params=["sum", "max", "min", "mean", "prod", "std", "var", "median", "kurt", "skew"]
)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names
    """
    return request.param


@pytest.fixture(params=["all", "any"])
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names
    """
    return request.param


# only == and != are support for GeometryArray
@pytest.fixture(params=["__eq__", "__ne__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations
    * ==
    * !=
    """
    return request.param


# -----------------------------------------------------------------------------
# Inherited tests
# -----------------------------------------------------------------------------

class TestInterface(extension_tests.BaseInterfaceTests):
    pass


class TestConstructors(extension_tests.BaseConstructorsTests):
    pass


class TestReshaping(extension_tests.BaseReshapingTests):
    pass


class TestGetitem(extension_tests.BaseGetitemTests):
    pass


class TestSetitem(extension_tests.BaseSetitemTests):
    pass


class TestMissing(extension_tests.BaseMissingTests):
    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]
        ser = pd.Series(data_missing)

        result = ser.fillna(fill_value)
        expected = pd.Series(data_missing._from_sequence([fill_value, fill_value]))
        self.assert_series_equal(result, expected)

        ## filling with array-like not yet supported

        # Fill with a series
        # result = ser.fillna(expected)
        # self.assert_series_equal(result, expected)

        ## Fill with a series not affecting the missing values
        # result = ser.fillna(ser)
        # self.assert_series_equal(result, ser)

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_limit_pad(self, data_missing):
        pass

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_limit_backfill(self, data_missing):
        pass

    @pytest.mark.skip("fillna method not supported")
    def test_fillna_series_method(self, data_missing, method):
        pass


class TestComparisonOps(extension_tests.BaseComparisonOpsTests):
    def check_opname(self, s, op_name, other, exc=None):
        # overwriting to indicate ops don't raise an error
        super().check_opname(s, op_name, other, exc=None)

    def _compare_other(self, s, data, op_name, other):
        self.check_opname(s, op_name, other)

    def test_compare_scalar(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)
        other = data[0]
        self._compare_other(s, data, op_name, other)




if __name__ == "__main__":
    def data_1():
        """Length-100 array for this type.

       * data[0] and data[1] should both be non missing
       * data[0] and data[1] should not be equal
       """

        wkt = [make_point(x, x) for x in range(100)]
        ga = from_wkt(wkt)
        return ga


    import numpy as np

    na_value = None
    data = data_1()


    a = data[:3]
    b = data[2:5]
    r1, r2 = pd.Series(a).align(pd.Series(b, index=[1, 2, 3]))

    # Assumes that the ctor can take a list of scalars of the type
    e1 = pd.Series(data._from_sequence(list(a) + [na_value], dtype=data.dtype))
    e2 = pd.Series(data._from_sequence([na_value] + list(b), dtype=data.dtype))
    extension_tests.BaseReshapingTests.assert_series_equal(r1, e1)


