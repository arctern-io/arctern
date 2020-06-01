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

from collections.abc import Iterable
from distutils.version import LooseVersion
import numbers
import pyarrow
from pandas.api.extensions import ExtensionDtype, ExtensionArray
from pandas.api.extensions import register_extension_dtype
import numpy as np
import pandas as pd
import arctern


@register_extension_dtype
class GeoDtype(ExtensionDtype):
    type = object
    name = "GeoDtype"
    na_value = pd.NA
    kind = 'O'

    def __repr__(self):
        return "GeoDtype"

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(type(string))
            )
        if string == cls.name:
            return cls()

        raise TypeError(
            "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
        )

    @classmethod
    def construct_array_type(cls):
        return GeoArray


def from_wkt(data):
    """
    Convert a list or array of wkt formed string to a GeoArray.
    :param data: array-like
            list or array of wkt formed string
    :return: GeoArray
    """
    return GeoArray(arctern.ST_GeomFromText(data).values)


def to_wkt(data):
    """
    Convert GeoArray or np.ndarray or list to a numpy string array of wkt formed string.
    """
    if not isinstance(data, (GeoArray, np.ndarray, list)):
        raise ValueError("'data' must be a GeoArray or np.ndarray or list.")
    return np.asarray(arctern.ST_AsText(data), dtype=str)


def from_wkb(data):
    """
    Convert a list or array of wkb objects to a GeoArray.
    :param data: array-like
            list or array of wkb objects
    :return: GeoArray
    """
    # pandas.infer_type can't infer custom ExtensionDtype
    if not isinstance(getattr(data, "dtype", None), GeoDtype) and len(data) != 0:
        from pandas.api.types import infer_dtype
        inferred = infer_dtype(data, skipna=True)
        if inferred in ("bytes", "empty"):
            pass
        else:
            raise ValueError("'data' must be bytes type array or list.")
    if not isinstance(data, np.ndarray):
        array = np.empty(len(data), dtype=object)
        array[:] = data
    else:
        array = data

    mask = pd.isna(array)
    array[mask] = None
    return GeoArray(array)


def is_geometry_array(data):
    """
    Check if the data is array like, and dtype is `GeoDtype`.
    """
    return isinstance(getattr(data, "dtype", None), GeoDtype)


def is_scalar_geometry(data):
    """
    Check if the data is bytes dtype.
    """
    return isinstance(data, bytes)


class GeoArray(ExtensionArray):
    _dtype = GeoDtype()

    def __init__(self, data):
        if not isinstance(data, (np.ndarray, GeoArray)):
            raise TypeError(
                "'data' should be array of wkb formed bytes. Use from_wkt to construct a GeoArray.")
        if not data.ndim == 1:
            raise ValueError("'data' should be 1-dim array of wkb formed bytes.")

        self.data = data

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self.shape[0]

    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return self.data.nbytes

    def copy(self):
        return GeoArray(self.data.copy())

    def isna(self):
        return np.array([g is None or g is np.nan for g in self.data], dtype=bool)

    def fillna(self, value=None, method=None, limit=None):
        from pandas.util._validators import validate_fillna_kwargs
        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()
        from pandas.api.types import is_array_like, infer_dtype
        if is_array_like(value):
            if len(value) != len(self):
                raise ValueError(
                    f"Length of 'value' does not match. Got ({len(value)}) "
                    f"expected {len(self)}"
                )
            value = value[mask]
        else:
            # because pandas infer_type(scalar) cant work on scalar value, we put the value into a list
            value = [value]
        if mask.any():
            if method is not None:
                from pandas.core.missing import pad_1d
                from pandas.core.missing import backfill_1d
                func = pad_1d if method == "pad" else backfill_1d
                new_values = func(self.astype(object), limit=limit, mask=mask)
                new_values = self._from_sequence(new_values, dtype=self.dtype)
                # raise NotImplementedError("not support fillna with method")
            else:
                # translate value
                if not isinstance(getattr(value, "dtype", value), (GeoDtype, type(None))):
                    inferred_type = infer_dtype(value, skipna=True)
                    if inferred_type == "string":
                        value = arctern.ST_GeomFromText(value)
                    elif inferred_type == "bytes":
                        pass
                    else:
                        raise ValueError(
                            "can only fillna with wkt formed string or wkb formed bytes")

                # fill with value
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    def _bin_op(self, other, op):
        def convert_values(values):
            if isinstance(values, ExtensionArray) or pd.api.types.is_list_like(values):
                return values
            return np.array([values] * len(self), dtype=object)

        if isinstance(other, (pd.Series, pd.Index)):
            # rely on pandas to unbox and dispatch to us
            return NotImplemented

        lvalue = self.data
        rvalue = convert_values(other)
        if not (len(lvalue)) == len(rvalue):
            raise ValueError("Length between compare doesn't match.")

        rst = op(lvalue, rvalue)
        rst = np.asarray(rst, dtype=bool)
        return rst

    def __eq__(self, other):
        import operator
        return self._bin_op(other, operator.eq)

    def __ne__(self, other):
        import operator
        return self._bin_op(other, operator.ne)

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = None
            elif not is_scalar_geometry(fill_value):
                raise TypeError("Expected None or geometry fill value")
        result = take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value)
        if allow_fill and fill_value is None:
            result[pd.isna(result)] = None
        return GeoArray(result)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        # array-like, slice
        if str(pd.__version__) >= LooseVersion("0.26.0.dev"):
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # to numpy array, pass-through non-array-like indexers
            item = pd.api.indexers.check_array_indexer(self, item)

        if isinstance(item, (Iterable, slice)):
            return GeoArray(self.data[item])
        raise TypeError("Index type not supported ", type(item))

    def __setitem__(self, key, value):
        if str(pd.__version__) >= LooseVersion("0.26.0.dev"):
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # keys to numpy array, pass-through non-array-like indexers
            key = pd.api.indexers.check_array_indexer(self, key)

        if isinstance(key, np.ndarray) and key.dtype == bool:
            if not key.any():
                return

        scalar_key = pd.api.types.is_scalar(key)
        scalar_value = pd.api.types.is_scalar(value)
        if scalar_key and not scalar_value:
            raise ValueError("setting an array element with a sequence.")

        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, (list, np.ndarray)):
            value = from_wkb(value)
        if isinstance(value, GeoArray):
            self.data[key] = value.data

        elif isinstance(value, bytes):
            self.data[key] = value
        elif pd.isna(value):
            self.data[key] = None
        else:
            raise TypeError("Value must be bytes value, got %s" % str(value))

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return from_wkb(scalars)

    def _values_for_factorize(self):
        # we store geometry as bytes internally, just return it.
        return self.data, None

    @classmethod
    def _from_factorized(cls, values, original):
        return GeoArray(values)

    def __arrow_array__(self, type=None):
        # convert the underlying array values to a pyarrow Array
        return pyarrow.array(self.data, type=type)

    def __array__(self, dtype=None):
        """
        The numpy array interface.

        Returns
        -------
        values : numpy array
        """
        return self.data

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple array

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray
        """
        data = np.concatenate([ga.data for ga in to_concat])
        return GeoArray(data)

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, GeoDtype):
            if copy:
                return self.copy()
            return self
        # as str or string type means to wkt formed string
        if pd.api.types.is_dtype_equal(dtype, str) or pd.api.types.is_dtype_equal(dtype, 'string'):
            return to_wkt(self)
        # TODO(shengjh): Currently we can not handle dtype which is not numpy.dtype, if get here
        return np.array(self, dtype=dtype, copy=copy)

    def _formatter(self, boxed=False):
        """Formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """
        if boxed:
            return lambda x: to_wkt([x])[0]
        return repr
