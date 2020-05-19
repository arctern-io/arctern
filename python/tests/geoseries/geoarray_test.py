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

"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite (by inheriting the pandas test suite), and should
contain no other tests.

The tests in this file are inherited from the BaseExtensionTests provided by pandas,
and only minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).

see https://pandas.pydata.org/pandas-docs/stable/development/extending.html#testing-extension-arrays
for more info.
"""

# pylint: disable=redefined-outer-name

import pandas as pd
from pandas.tests.extension import base
import pytest
from arctern.geoseries.geoarray import GeoDtype, from_wkt


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
def data_missing():
    """Length-2 array with [NA, Valid]"""
    ga = from_wkt(["invalid geo", make_point(1, 1)])
    return ga


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    if request.param == "data_missing":
        return data_missing
    return None


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


# only == and != are support for GeoArray
@pytest.fixture(params=["__eq__", "__ne__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations
    * ==
    * !=
    """
    return request.param


@pytest.fixture(params=[
    "__add__",
    "__radd__",
    '__sub__',
    '__rsub__',
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
])
def all_arithmetic_operators(request):
    return request.param


_all_numeric_reductions = [
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """
    Fixture for numeric reduction names.
    """
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """
    Fixture for boolean reduction names.
    """
    return request.param


class TestDtype(base.BaseDtypeTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestReshaping(base.BaseReshapingTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestSetitem(base.BaseSetitemTests):
    pass


class TestMissing(base.BaseMissingTests):
    pass


class TestReduce(base.BaseNoReduceTests):
    """ we don't define any reductions """


class TestComparisonOps(base.BaseComparisonOpsTests):
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


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    @pytest.mark.skip("not implemented")
    def test_divmod_series_array(self, data, data_for_twos):
        pass

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        pass

    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # ndarray & other series
        exc = self.series_array_exc
        op_name = all_arithmetic_operators
        # TODO(shengjh): how to solve this?
        # because we can't prevent 'op(GeoArray,np.array)' when op
        # is '__add__' or '__radd__'.
        if op_name in ['__add__', '__radd__']:
            pass
        else:
            s = pd.Series(data)
            self.check_opname(s, op_name, pd.Series([s.iloc[0]] * len(s)), exc=exc)

    def test_add_series_with_extension_array(self, data):
        op_name = '__add__'
        s = pd.Series(data)
        self.check_opname(s, op_name, data, exc=TypeError)


class TestMethods(base.BaseMethodsTests):
    @pytest.mark.skip("not support value counts")
    def test_value_counts(self, all_data, dropna):
        pass

    @pytest.mark.skip("not support sort")
    def test_sort_values(self, data_for_sorting, ascending):
        pass

    @pytest.mark.skip("not support sort")
    def test_sort_values_missing(self, data_missing_for_sorting, ascending):
        pass

    @pytest.mark.skip("not support sort")
    def test_argsort(self, data_for_sorting):
        pass

    @pytest.mark.skip("not support sort")
    def test_nargsort(self, data_missing_for_sorting, na_position, expected):
        pass

    @pytest.mark.skip("not support sort")
    def test_sort_values_frame(self, data_for_sorting, ascending):
        pass

    @pytest.mark.skip("not support sort")
    def test_argsort_missing(self, data_missing_for_sorting):
        pass

    @pytest.mark.skip("not support sort")
    def test_argsort_missing_array(self, data_missing_for_sorting):
        pass

    @pytest.mark.skip("not support sort")
    def test_searchsorted(self, data_for_sorting, as_series):
        pass


class TestGroupby(base.BaseGroupbyTests):
    @pytest.mark.skip("not support sort")
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        pass


class TestCasting(base.BaseCastingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


@pytest.mark.skip("not implemented")
class TestParsing(base.BaseParsingTests):
    pass
