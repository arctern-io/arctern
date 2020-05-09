import pyarrow
from pandas.api.extensions import ExtensionDtype, ExtensionArray
from pandas.api.extensions import register_extension_dtype
from collections.abc import Iterable
import numpy as np
import pandas as pd
import arctern
import numbers
from distutils.version import LooseVersion


@register_extension_dtype
class GeoDtype(ExtensionDtype):
    type = bytes
    name = "GeoDtype"
    na_value = None
    kind = 'O'

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(type(string))
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
            )

    @classmethod
    def construct_array_type(cls):
        return GeoArray


def from_wkt(data, crs=None):
    """
    Convert a list or array of wkt formed string to a GeoArray.
    :param data: array-like
            list or array of wkt formed string
    :param crs: string, optional
            Coordinate Reference System of the geometry objects. such as an authority string (eg "EPSG:4326").
    :return: GeoArray
    """
    # TODO(shengjh): support shaply wkt object?
    return GeoArray(arctern.ST_GeomFromText(data).values, crs)


def is_geometry_arry(data):
    """
    Check if the data is geometry array like GeoArray, GeoSeries or Series[GeoArray].
    """
    if isinstance(getattr(data, "dtype", None), GeoDtype):
        # GeoArray, GeoSeries and Series[GeoArray]
        return True
    else:
        return False


def is_scalar_geometry(data):
    """
    Check if the data is of bytes dtype.
    """
    return isinstance(data, bytes)


class GeoArray(ExtensionArray):
    _dtype = GeoDtype()

    def __init__(self, data, crs=None):
        if isinstance(data, self.__class__):
            if not crs and hasattr(data, "crs"):
                crs = data.crs
            data = data
        elif not isinstance(data, np.ndarray):
            raise TypeError("'data' should be array of wkb formed bytes. Use from_wkt to construct a GeoArray.")
        elif not data.ndim == 1:
            raise ValueError(
                "'data' should be 1-dim array of wkb formed bytes"
            )

        self.data = data
        self._crs = None
        self.crs = crs

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        # self._crs = None if not value else CRS.from_user_input(value)
        self._crs = value

    @property
    def dtype(self) -> ExtensionDtype:
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

    def nbytes(self):
        return self.data.nbytes

    def copy(self):
        return GeoArray(self.data.copy())

    def isna(self):
        return np.array([g is None for g in self.data])

    def _bin_op(self, other, op):
        def convert_values(values):
            if isinstance(values, ExtensionArray) or pd.api.types.is_list_like(values):
                return values
            else:
                return [values] * len(self)

        lvalue = self.data
        rvalue = convert_values(other)
        if not (len(lvalue)) == len(rvalue):
            raise ValueError("Length between compare doesn't match.")
        # artern python api can receive any type data,
        # which can received by pyarrow.array(sequence, iterable, ndarray or Series)
        rst = op(lvalue, rvalue)
        rst = np.asarray(rst)
        return rst

    def __eq__(self, other):
        rst = self._bin_op(other, arctern.ST_Equals)
        return rst

    def __ne__(self, other):
        return ~self.__eq__(other)

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
        return GeoArray(result, crs=self.crs)

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        # array-like, slice
        if isinstance(item, (Iterable, slice)):
            return GeoArray(self.data[item])
        else:
            raise TypeError("Index type not supported", item)

    def __setitem__(self, key, value):
        # TODO: check this
        if str(pd.__version__) >= LooseVersion("0.26.0.dev"):
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # keys to numpy array, pass-through non-array-like indexers
            key = pd.api.indexers.check_array_indexer(self, key)

        if isinstance(value, (list, np.ndarray)):
            value = GeoArray(value)
        if isinstance(value, GeoArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("Cannot set a single element with an array")
            self.data[key] = value.data

        if isinstance(value, bytes) or value is None:
            value = value
            if isinstance(key, (slice, list, np.ndarray)):
                value_arry = np.empty(1, dtype=bytes)
                value_arry[:] = [value]
                self.data[key] = value_arry
            else:
                self.data[key] = value
        else:
            raise TypeError("Value should be wkb formed bytes or None, got %s" % str(value))

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return GeoArray(scalars)

    def __arrow_array__(self, type=None):
        # convert the underlying array values to a pyarrow Array
        return pyarrow.array(self.data, type=type)
