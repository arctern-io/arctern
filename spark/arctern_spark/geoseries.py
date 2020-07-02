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

# pylint: disable=protected-access,too-many-public-methods,too-many-branches,unidiomatic-typecheck


import pandas as pd
from pandas.io.formats.printing import pprint_thing
import databricks.koalas as ks
from databricks.koalas import DataFrame, Series, get_option
from databricks.koalas.exceptions import SparkPandasIndexingError
from databricks.koalas.internal import NATURAL_ORDER_COLUMN_NAME
from databricks.koalas.series import REPR_PATTERN
from databricks.koalas.utils import (
    validate_axis,
    validate_bool_kwarg,
)
from pyspark.sql import functions as F, Column
from pyspark.sql.window import Window
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    BinaryType,
)

from . import scala_wrapper


# for unary or binary operation, which return koalas Series.
def _column_op(f, *args):
    from . import scala_wrapper
    return ks.base.column_op(getattr(scala_wrapper, f))(*args)


# for unary or binary operation, which return GeoSeries.
def _column_geo(f, *args, **kwargs):
    from . import scala_wrapper
    kss = ks.base.column_op(getattr(scala_wrapper, f))(*args)
    return GeoSeries(kss, **kwargs)


def _agg(f, kss):
    scol = getattr(scala_wrapper, f)(kss.spark.column)
    sdf = kss._internal._sdf.select(scol)
    kdf = sdf.to_koalas()
    return GeoSeries(first_series(kdf), crs=kss._crs)


def _validate_crs(crs):
    if crs is not None and not isinstance(crs, str):
        raise TypeError("`crs` should be spatial reference identifier string")
    crs = crs.upper() if crs is not None else crs
    return crs


def _validate_arg(arg):
    if isinstance(arg, str):
        arg = F.lit(arg)
        arg = getattr(scala_wrapper, "st_geomfromtext")(arg)
    elif isinstance(arg, (bytearray, bytes)):
        arg = F.lit(arg)
        arg = getattr(scala_wrapper, "st_geomfromwkb")(arg)
    elif not isinstance(arg, Series):
        arg = Series(arg)
    return arg


def _validate_args(*args, dtype=None):
    series_length = 1
    for arg in args:
        if not isinstance(arg, dtype):
            if series_length < len(arg):
                series_length = len(arg)
    args_list = []
    for i, arg in enumerate(args):
        if isinstance(arg, dtype):
            if i == 0:
                args_list.append(Series([arg] * series_length))
            else:
                args_list.append(F.lit(arg))
        elif not isinstance(arg, Series):
            args_list.append(Series(arg))
        else:
            args_list.append(arg)
    return args_list


class GeoSeries(Series):
    def __init__(
            self, data=None, index=None, dtype=None, name=None, copy=False, crs=None, fastpath=False
    ):
        if isinstance(data, DataFrame):
            assert index is not None
            assert dtype is None
            assert name is None
            assert not copy
            assert not fastpath
            anchor = data
            column_label = index
        else:
            if isinstance(data, pd.Series):
                assert index is None
                assert dtype is None
                assert name is None
                assert not copy
                assert not fastpath
                s = data
            elif isinstance(data, ks.Series):
                assert index is None
                assert dtype is None
                assert name is None
                if hasattr(data, "crs") and crs:
                    if not data.crs == crs and data.crs is not None:
                        raise ValueError(
                            "crs of the passed geometry data is different from crs.")

                s = data
            else:
                s = pd.Series(
                    data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath
                )
                import numpy as np
                # The default dtype for empty Series is 'float64' in pandas, but it will be object in future.
                # see https://github.com/pandas-dev/pandas/pull/29405
                if s.empty and (s.dtype == np.dtype("float64") or s.dtype == np.dtype("object")):
                    # we can't create an empty pandas series, which dtype can be infered as
                    # pyspark StringType or Binary Type, so we create a koalas empty series
                    # and cast it's type to StringType
                    s = Series([], dtype=int).astype(str)

            anchor = DataFrame(s)
            column_label = anchor._internal.column_labels[0]
            kss = anchor._kser_for(column_label)

            spark_dtype = kss.spark.data_type
            if isinstance(spark_dtype, scala_wrapper.GeometryUDT):
                pass
            elif isinstance(spark_dtype, BinaryType):
                kss = _column_op("st_geomfromwkb", kss)
            elif isinstance(spark_dtype, StringType):
                kss = _column_op("st_geomfromtext", kss)
            else:
                raise TypeError(
                    "Can not use no StringType or BinaryType or GeometryUDT data to construct GeoSeries.")
            anchor = kss._kdf
            anchor._kseries = {column_label: kss}

        super(Series, self).__init__(anchor)
        self._column_label = column_label
        self.set_crs(crs)

    def __getitem__(self, key):
        try:
            if (isinstance(key, slice) and any(type(n) == int for n in [key.start, key.stop])) or (
                type(key) == int
                and not isinstance(self.index.spark.data_type, (IntegerType, LongType))
            ):
                # Seems like pandas Series always uses int as positional search when slicing
                # with ints, searches based on index values when the value is int.
                r = self.iloc[key]
                return GeoSeries(r, crs=self.crs) if isinstance(r, Series) else r
            r = self.loc[key]
            return GeoSeries(r, crs=self.crs) if isinstance(r, Series) else r
        except SparkPandasIndexingError:
            raise KeyError(
                "Key length ({}) exceeds index depth ({})".format(
                    len(key), len(self._internal.index_map)
                )
            )

    def set_crs(self, crs):
        """
        Sets the Coordinate Reference System (CRS) for all geometries in GeoSeries.

        Parameters
        ----------
        crs : str
            A string representation of CRS.
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, "EPSG:4326".

        Notes
        -------
        Arctern supports common CRSs listed at the `Spatial Reference <https://spatialreference.org/>`_ website.

        Examples
        -------
        >>> from arctern_pyspark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT(1 2)"])
        >>> s.set_crs("EPSG:4326")
        >>> s.crs
        'EPSG:4326'
        """
        crs = _validate_crs(crs)
        self._crs = crs

    @property
    def crs(self):
        """
        Returns the Coordinate Reference System (CRS) of the GeoSeries.

        Returns
        -------
        str
            CRS of the GeoSeries.

        Examples
        -------
        >>> from arctern_pyspark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT(1 2)"], crs="EPSG:4326")
        >>> s.crs
        'EPSG:4326'
        """
        return self._crs

    @crs.setter
    def crs(self, crs):
        """
        Sets the Coordinate Reference System (CRS) for all geometries in GeoSeries.

        Parameters
        ----------
        crs : str
            A string representation of CRS.
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, "EPSG:4326".

        Examples
        -------
        >>> from arctern_pyspark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT(1 2)"])
        >>> s.set_crs("EPSG:4326")
        >>> s.crs
        'EPSG:4326'
        """
        self.set_crs(crs)

    @property
    def hasnans(self):
        """
        Return True if it has any missing values. Otherwise, it returns False.
        """
        sdf = self._internal.spark_frame
        scol = self.spark.column

        return sdf.select(F.max(scol.isNull())).collect()[0][0]

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            wkt_ks = _column_op("st_astext", self)
            return wkt_ks._to_internal_pandas().to_string(name=self.name, dtype=self.dtype)

        wkt_ks = _column_op("st_astext", self.head(max_display_count + 1))
        pser = wkt_ks._to_internal_pandas()
        pser_length = len(pser)
        pser = pser.iloc[:max_display_count]
        if pser_length > max_display_count:
            repr_string = pser.to_string(length=True)
            rest, prev_footer = repr_string.rsplit("\n", 1)
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length = match.group("length")
                name = str(self.dtype.name)
                footer = "\nName: {name}, dtype: {dtype}\nShowing only the first {length}".format(
                    length=length, name=self.name, dtype=pprint_thing(name)
                )
                return rest + footer
        return pser.to_string(name=self.name, dtype=self.dtype)

    def _with_new_scol(self, scol):
        """
        Copy Koalas Series with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied Series
        """
        internal = self._kdf._internal.copy(
            column_labels=[self._column_label], data_spark_columns=[scol]
        )
        return first_series(DataFrame(internal))

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None):
        """Fill NA/NaN values.

        .. note:: the current implementation of 'method' parameter in fillna uses Spark's Window
            without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        value : scalar, dict, Series
            Value to use to fill holes. alternately a dict/Series of values
            specifying which value to use for each column.
            DataFrame is not supported.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series pad / ffill: propagate last valid
            observation forward to next valid backfill / bfill:
            use NEXT valid observation to fill gap
        axis : {0 or `index`}
            1 and `columns` are not supported.
        inplace : boolean, default False
            Fill in place (do not create a new object)
        limit : int, default None
            If method is specified, this is the maximum number of consecutive NaN values to
            forward/backward fill. In other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None

        Returns
        -------
        Series
            Series with NA entries filled.
        """
        return self._fillna(value, method, axis, inplace, limit)

    def _fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, part_cols=()):
        axis = validate_axis(axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        if axis != 0:
            raise NotImplementedError(
                "fillna currently only works for axis=0 or axis='index'")
        if (value is None) and (method is None):
            raise ValueError(
                "Must specify a fillna 'value' or 'method' parameter.")
        if (method is not None) and (method not in ["ffill", "pad", "backfill", "bfill"]):
            raise ValueError(
                "Expecting 'pad', 'ffill', 'backfill' or 'bfill'.")

        scol = self.spark.column
        cond = scol.isNull()

        if value is not None:
            if not isinstance(value, (str, bytearray, bytes)):
                raise TypeError("Unsupported type %s" % type(value))
            if limit is not None:
                raise ValueError(
                    "limit parameter for value is not support now")
            value = _validate_arg(value)
            scol = F.when(cond, value).otherwise(scol)
        else:
            if method in ["ffill", "pad"]:
                func = F.last
                end = Window.currentRow - 1
                if limit is not None:
                    begin = Window.currentRow - limit
                else:
                    begin = Window.unboundedPreceding
            elif method in ["bfill", "backfill"]:
                func = F.first
                begin = Window.currentRow + 1
                if limit is not None:
                    end = Window.currentRow + limit
                else:
                    end = Window.unboundedFollowing

            window = (
                Window.partitionBy(*part_cols)
                .orderBy(NATURAL_ORDER_COLUMN_NAME)
                .rowsBetween(begin, end)
            )
            scol = F.when(cond, func(scol, True).over(window)).otherwise(scol)

        if inplace:
            self._kdf._update_internal_frame(
                self._kdf._internal.with_new_spark_column(
                    self._column_label, scol)
            )
        else:
            return self._with_new_scol(scol).rename(self.name)

    def __eq__(self, other):
        return ks.base.column_op(Column.__eq__)(self, _validate_arg(other))

    def __ne__(self, other):
        return ks.base.column_op(Column.__ne__)(self, _validate_arg(other))

    # -------------------------------------------------------------------------
    # Geometry related property
    # -------------------------------------------------------------------------

    @property
    def area(self):
        return _column_op("st_area", self)

    @property
    def is_valid(self):
        return _column_op("st_isvalid", self)

    @property
    def length(self):
        return _column_op("st_length", self)

    @property
    def is_simple(self):
        return _column_op("st_issimple", self)

    @property
    def geom_type(self):
        return _column_op("st_geometrytype", self)

    @property
    def centroid(self):
        return _column_geo("st_centroid", self, crs=self._crs)

    @property
    def convex_hull(self):
        return _column_geo("st_convexhull", self, crs=self._crs)

    @property
    def npoints(self):
        return _column_op("st_npoints", self)

    @property
    def envelope(self):
        return _column_geo("st_envelope", self, crs=self._crs)

    @property
    def exterior(self):
        return _column_geo("st_exteriorring", self)

    @property
    def is_empty(self):
        return _column_op("st_isempty", self)

    @property
    def boundary(self):
        return _column_geo("st_boundary", self)

    # -------------------------------------------------------------------------
    # Geometry related unary methods, which return GeoSeries
    # -------------------------------------------------------------------------
    def make_valid(self):
        return _column_geo("st_makevalid", self, crs=self._crs)

    def precision_reduce(self, precision):
        return _column_geo("st_precisionreduce", self, F.lit(precision), crs=self._crs)

    def unary_union(self):
        return _agg("st_union_aggr", self)

    def envelope_aggr(self):
        return _agg("st_envelope_aggr", self)

    def curve_to_line(self):
        return _column_geo("st_curvetoline", self, crs=self._crs)

    def simplify(self, tolerance):
        return _column_geo("st_simplifypreservetopology", self, F.lit(tolerance), crs=self._crs)

    def buffer(self, distance):
        return _column_geo("st_buffer", self, F.lit(distance), crs=self._crs)

    def to_crs(self, crs):
        """
        Transforms the Coordinate Reference System (CRS) of the GeoSeries to `crs`.

        Parameters
        ----------
        crs : str
            A string representation of CRS.
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, ``"EPSG:4326"``.

        Returns
        -------
        GeoSeries
            GeoSeries with transformed CRS.

        Notes
        -------
        Arctern supports common CRSs listed at the `Spatial Reference <https://spatialreference.org/>`_ website.

        Examples
        -------
        >>> from arctern_pyspark import GeoSeries
        >>> s = GeoSeries(["POINT (1 2)"], crs="EPSG:4326")
        >>> s
        0    POINT (1 2)
        dtype: GeoDtype
        >>> s.to_crs(crs="EPSG:3857")
        0    POINT (111319.490793274 222684.208505545)
        dtype: GeoDtype
        """
        if crs is None:
            raise ValueError("Can not transform with invalid crs")
        if self.crs is None:
            raise ValueError(
                "Can not transform geometries without crs. Set crs for this GeoSeries first.")
        if self.crs == crs:
            return self
        return _column_geo("st_transform", self, F.lit(self.crs), F.lit(crs), crs=crs)

    def scale(self, factor_x, factor_y):
        return _column_geo("st_scale", self, F.lit(factor_x), F.lit(factor_y))

    def affine(self, a, b, d, e, offset_x, offset_y):
        return _column_geo("st_affine", self, F.lit(a), F.lit(b), F.lit(d), F.lit(e), F.lit(offset_x), F.lit(offset_y))

    def translate(self, shifter_x, shifter_y):
        return _column_geo("st_translate", self, F.lit(shifter_x), F.lit(shifter_y))

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return Series[bool/float]
    # -------------------------------------------------------------------------

    def intersects(self, other):
        return _column_op("st_intersects", self, _validate_arg(other))

    def within(self, other):
        return _column_op("st_within", self, _validate_arg(other))

    def contains(self, other):
        return _column_op("st_contains", self, _validate_arg(other))

    def geom_equals(self, other):
        return _column_op("st_equals", self, _validate_arg(other))

    def crosses(self, other):
        return _column_op("st_crosses", self, _validate_arg(other))

    def touches(self, other):
        return _column_op("st_touches", self, _validate_arg(other))

    def overlaps(self, other):
        return _column_op("st_overlaps", self, _validate_arg(other))

    def distance(self, other):
        return _column_op("st_distance", self, _validate_arg(other))

    def distance_sphere(self, other):
        return _column_op("st_distancesphere", self, _validate_arg(other))

    def hausdorff_distance(self, other):
        return _column_op("st_hausdorffdistance", self, _validate_arg(other))

    def disjoint(self, other):
        return _column_op("st_disjoint", self, _validate_arg(other))

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def intersection(self, other):
        return _column_geo("st_intersection", self, _validate_arg(other), crs=self.crs)

    def difference(self, other):
        return _column_geo("st_difference", self, _validate_arg(other))

    def symmetric_difference(self, other):
        return _column_geo("st_symdifference", self, _validate_arg(other))

    def union(self, other):
        return _column_geo("st_union", self, _validate_arg(other))

    # -------------------------------------------------------------------------
    # Geometry related quaternary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def rotate(self, rotation_angle, origin, use_radians=False):
        import math
        if not use_radians:
            rotation_angle = rotation_angle * math.pi / 180.0

        if origin is None:
            return _column_geo("st_rotate", self, F.lit(rotation_angle))
        elif isinstance(origin, str):
            return _column_geo("st_rotate", self, F.lit(rotation_angle), F.lit(origin))
        elif isinstance(origin, tuple):
            origin_x = origin[0]
            origin_y = origin[1]
            return _column_geo("st_rotate", self, F.lit(rotation_angle), F.lit(origin_x), F.lit(origin_y))

    # -------------------------------------------------------------------------
    # utils
    # -------------------------------------------------------------------------

    @classmethod
    def polygon_from_envelope(cls, min_x, min_y, max_x, max_y, crs=None):
        dtype = (float, int)
        min_x, min_y, max_x, max_y = _validate_args(
            min_x, min_y, max_x, max_y, dtype=dtype)
        _kdf = ks.DataFrame(min_x)
        kdf = _kdf.rename(columns={_kdf.columns[0]: "min_x"})
        kdf["min_y"] = min_y
        kdf["max_x"] = max_x
        kdf["max_y"] = max_y
        return _column_geo("st_polygonfromenvelope", kdf["min_x"], kdf["min_y"], kdf["max_x"], kdf["max_y"], crs=crs)

    @classmethod
    def point(cls, x, y, crs=None):
        dtype = (float, int)
        return _column_geo("st_point", *_validate_args(x, y, dtype=dtype), crs=crs)

    @classmethod
    def geom_from_geojson(cls, json, crs=None):
        return _column_geo("st_geomfromgeojson", _validate_arg(json), crs=crs)

    def as_geojson(self):
        return _column_op("st_asgeojson", self)

    def to_wkt(self):
        return _column_op("st_astext", self)

    def to_wkb(self):
        return _column_op("st_aswkb", self)

    def head(self, n: int = 5):
        r = super().head(n)
        r.set_crs(self.crs)
        return r

    def take(self, indices):
        r = super().take(indices)
        r.set_crs(self.crs)
        return r


def first_series(df):
    """
    Takes a DataFrame and returns the first column of the DataFrame as a Series
    """
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    if isinstance(df, DataFrame):
        kss = df._kser_for(df._internal.column_labels[0])
        if isinstance(kss.spark.data_type, scala_wrapper.GeometryUDT):
            return GeoSeries(kss)
        return kss
    return df[df.columns[0]]


ks.series.first_series = first_series
