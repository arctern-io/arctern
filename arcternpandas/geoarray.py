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
    type = object
    name = "GeoDtype"
    na_value = np.nan
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


def to_wkt(data):
    """
    Convert GeoArray or np.ndarray or list to a numpy object array of wkt formed string.
    """
    if not isinstance(data, (GeoArray, np.ndarray, list)):
        raise ValueError("'data' must be a GeoArray or np.ndarray or list.")
    return arctern.ST_AsText(data).values


def from_wkb(data, crs=None):
    """
    Convert a list or array of wkb objects to a GeoArray.
    :param data: array-like
            list or array of wkb objects
    :param crs: string optional
            Coordinate Reference System of the geometry objects. such as an authority string (eg "EPSG:4326").
    :return: GeoArray
    """
    first_invalid = None
    for item in data:
        if item is not None or item is not np.nan:
            first_invalid = item
            break
    if first_invalid is not None:
        if not isinstance(first_invalid, bytes):
            raise ValueError("'data' must be bytes type array or list.")

    if not isinstance(data, np.ndarray):
        arr = np.empty(len(data), dtype=object)
        arr[:] = data
    else:
        arr = data

    return GeoArray(arr, crs=crs)


def is_geometry_arry(data):
    """
    Check if the data is geometry array like GeoArray, GeoSeries or Series[GeoArray].
    """
    # GeoArray, GeoSeries and Series[GeoArray]
    return isinstance(getattr(data, "dtype", None), GeoDtype)


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
            raise ValueError("'data' should be 1-dim array of wkb formed bytes.")

        self.data = data
        # TODO(shengjh): do we need crs in here?
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

    def nbytes(self):
        return self.data.nbytes

    def copy(self):
        return GeoArray(self.data.copy(), crs=self.crs)

    def isna(self):
        return np.array([g is None or g is np.nan for g in self.data], dtype=bool)

    def _bin_op(self, other, op):
        def convert_values(values):
            if isinstance(values, ExtensionArray) or pd.api.types.is_list_like(values):
                return values
            else:
                return [values] * len(self)

        if isinstance(other, (pd.Series, pd.Index)):
            # rely on pandas to unbox and dispatch to us
            return NotImplemented

        lvalue = self.data
        rvalue = convert_values(other)
        if not (len(lvalue)) == len(rvalue):
            raise ValueError("Length between compare doesn't match.")
        # artern python api can receive any type data,
        # which can received by pyarrow.array(sequence, iterable, ndarray or Series)
        rst = op(lvalue, rvalue)
        rst = np.asarray(rst, dtype=bool)
        return rst

    def __eq__(self, other):
        return self._bin_op(other, arctern.ST_Equals)

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
        if str(pd.__version__) >= LooseVersion("0.26.0.dev"):
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # to numpy array, pass-through non-array-like indexers
            item = pd.api.indexers.check_array_indexer(self, item)

        if isinstance(item, (Iterable, slice)):
            return GeoArray(self.data[item])
        else:
            raise TypeError("Index type not supported ", item)

    def __setitem__(self, key, value):
        if str(pd.__version__) >= LooseVersion("0.26.0.dev"):
            # for pandas >= 1.0, validate and convert IntegerArray/BooleanArray
            # keys to numpy array, pass-through non-array-like indexers
            key = pd.api.indexers.check_array_indexer(self, key)

        if isinstance(value, (list, np.ndarray)):
            value = from_wkb(value)
        if isinstance(value, GeoArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("Cannot set a single element with an array")
            self.data[key] = value.data

        elif isinstance(value, bytes) or value is None or value is np.nan:
            value = value
            if isinstance(key, (slice, list, np.ndarray)):
                value_arry = np.empty(1, dtype=object)
                value_arry[:] = [value]
                self.data[key] = value_arry
            else:
                self.data[key] = value
        else:
            raise TypeError("Value should be array-like bytes  or None, got %s" % str(value))

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return from_wkb(scalars)

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
        return GeoArray(data, crs=to_concat[0].crs)

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
            else:
                return self
        elif pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(dtype):
            # as string type means to wkt formed string
            return to_wkt(self)
        else:
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
