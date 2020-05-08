import pyarrow
from pandas.api.extensions import ExtensionDtype, ExtensionArray
from pandas.api.extensions import register_extension_dtype
from collections.abc import Iterable
from pyproj import CRS
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
        return GeoArry


def from_wkt(data, crs=None):
    """
            Convert a list or array of wkt formed string to a GeoArray.
    :param data: array-like
            list or array of wkt formed string
    :param crs: value, optional
            Coordinate Reference System of the geometry objects. Can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a wkt string.
    :return: GeoArry
    """
    # TODO(shengjh): may support shaply wkt object
    return GeoArry(arctern.ST_GeomFromText(data).values, crs)


class GeoArry(ExtensionArray):
    _dtype = GeoDtype()
    data: np.ndarray
    _crs: CRS

    def __init__(self, data, crs=None):
        if isinstance(data, self.__class__):
            if not crs and hasattr(data, "crs"):
                crs = data.crs
            data = data
        elif not isinstance(data, np.ndarray):
            raise TypeError("'data' should be arrry of wkb formed bytes. Use from_wkt to construct a GeoArry.")
        elif not data.ndim == 1:
            raise ValueError(
                "'data' should be 1-dim arry of wkb formed bytes"
            )

        self.data = data
        self.crs = crs

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        self._crs = None if not value else CRS.from_user_input(value)

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

    def isna(self):
        # return ~arctern.ST_IsValid(self.data).values.astype(bool)
        return np.array([g is None for g in self.data])

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        # array-like, slice
        if isinstance(item, (Iterable, slice)):
            return GeoArry(self.data[item])
        else:
            raise TypeError("Index type not supported", item)

    def __setitem__(self, key, value):
        if str(pd.__version__) >= LooseVersion("0.26.0.dev"):
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # keys to numpy array, pass-through non-array-like indexers
            key = pd.api.indexers.check_array_indexer(self, key)

        if isinstance(value, (list, np.ndarray)):
            value = GeoArry(value)
        if isinstance(value, GeoArry):
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
        return GeoArry(scalars)

    def __arrow_array__(self, type=None):
        # convert the underlying array values to a pyarrow Array
        return pyarrow.array(self.data, type=type)
