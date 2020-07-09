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

# pylint: disable=protected-access,too-many-public-methods,too-many-branches
# pylint: disable=super-init-not-called,unidiomatic-typecheck,unbalanced-tuple-unpacking
# pylint: disable=too-many-lines,non-parent-init-called

import databricks.koalas as ks
import numpy as np
import pandas as pd
from databricks.koalas import DataFrame, Series, get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.exceptions import SparkPandasIndexingError
from databricks.koalas.internal import NATURAL_ORDER_COLUMN_NAME
from databricks.koalas.series import REPR_PATTERN
from databricks.koalas.utils import (
    validate_axis,
    validate_bool_kwarg,
)
from pandas.api.types import is_list_like
from pandas.io.formats.printing import pprint_thing
from pyspark.sql import functions as F, Column
from pyspark.sql.types import (
    IntegerType,
    LongType,
    StringType,
    BinaryType,
)
from pyspark.sql.window import Window

from . import scala_wrapper


# for unary or binary operation, which return koalas Series.
def _column_op(f, *args):
    return ks.base.column_op(getattr(scala_wrapper, f))(*args)


# for unary or binary operation, which return GeoSeries.
def _column_geo(f, *args, **kwargs):
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
        arg = scala_wrapper.st_geomfromtext(F.lit(arg))
    elif isinstance(arg, (bytearray, bytes)):
        arg = scala_wrapper.st_geomfromwkb(F.lit(arg))
    elif isinstance(arg, Series):
        pass
    elif is_list_like(arg) or isinstance(pd.Series):
        arg = Series(arg)
    else:
        raise TypeError("Unsupported type %s" % type(arg))
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
        elif isinstance(arg, pd.Series):
            args_list.append(Series(arg))
        elif isinstance(arg, Series):
            args_list.append(arg)
        elif is_list_like(arg):
            args_list.append(Series(arg))
        else:
            raise TypeError("Unsupported type %s" % type(arg))
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
                if hasattr(data, "crs"):
                    if crs and data.crs and not data.crs == crs:
                        raise ValueError("crs of the passed geometry data is different from crs.")
                    crs = data.crs or crs
                s = data
            else:
                s = pd.Series(
                    data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath
                )
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

        IndexOpsMixin.__init__(self, anchor)
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
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT(1 2)"])
        >>> s.set_crs("EPSG:4326")
        >>> s.crs
        'EPSG:4326'
        """
        crs = _validate_crs(crs)
        self._crs = crs

        if hasattr(self, "_gdf") and self._gdf is not None:
            self._gdf._crs_for_cols[self.name] = self._crs

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
        >>> from arctern_spark import GeoSeries
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
        >>> from arctern_spark import GeoSeries
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

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT(1 2)", None])
        >>> s.hasnans
        True
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

        Parameters
        ----------
        scol: the new Spark Column

        Returns
        -------
        GeoSeries
            The copied GeoSeries
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
        GeoSeries
            GeoSeries with NA entries filled.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT(1 2)", None])
        >>> r = s.fillna(s[0])
        >>> r
        0    POINT (1 1)
        1    POINT (1 2)
        2    POINT (1 1)
        Name: 0, dtype: object
        """
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
            part_cols = ()
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
            return self
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
        """
        Calculates the 2D Cartesian (planar) area of each geometry in the GeoSeries.

        The ways to calculate the area of geometries are as follows:

        * POINT / MULTIPOINT / LINESTRING / MULTILINESTRING / CIRCULARSTRING: 0
        * POLYGON: Area of a single polygon.
        * MULTIPOLYGON: Sum of area of multiple polygons.
        * CURVEPOLYGON: Area of a single curvilinear polygon.
        * MULTICURVE: Sum of area of multiple curvilinear polygons.
        * MULTISURFACE / COMPOUNDCURVE / GEOMETRYCOLLECTION: For a geometry collection among the 3 types, calculates the sum of area of all geometries in the collection.

        Returns
        -------
        Koalas Series
            2D Cartesian (planar) area of each geometry in the GeoSeries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))", None])
        >>> s.area
        0    0.0
        1    4.0
        2    NaN
        Name: 0, dtype: float64
        """
        return _column_op("st_area", self)

    @property
    def is_valid(self):
        """
        Tests whether each geometry in the GeoSeries is in valid format, such as WKT and WKB formats.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether an element is valid.

            * *True:* The geometry is valid.
            * *False:* The geometry is invalid.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))", "POLYGON ((0 0 , 1 1, 1 0, 0 0))", "POINT (1)", None])
        >>> s.is_valid
        0     True
        1     True
        2     True
        3     True
        4    False
        Name: 0, dtype: bool
        """
        return _column_op("st_isvalid", self).astype(bool)

    @property
    def length(self):
        """
        Calculates the length of each geometry in the GeoSeries.

        The ways to calculate the length of geometries are as follows:

        * POINT / MULTIPOINT / POLYGON / MULTIPOLYGON / CURVEPOLYGON / MULTICURVE: 0
        * LINESTRING: Length of a single straight line.
        * MULTILINESTRING: Sum of length of multiple straight lines.
        * CIRCULARSTRING: Length of a single curvilinear line.
        * MULTISURFACE / COMPOUNDCURVE / GEOMETRYCOLLECTION: For a geometry collection among the 3 types, calculates the sum of length of all geometries in the collection.

        Returns
        -------
        Koalas Series
            Length of each geometry in the GeoSeries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT (1 1)", "LINESTRING (2 0, 2 2, 2 6)", "POLYGON ((3 3, 7 3, 7 7, 3 7, 3 3))"])
        >>> s.length
        0     0.0
        1     6.0
        2    16.0
        Name: 0, dtype: float64
        """
        return _column_op("st_length", self)

    @property
    def is_simple(self):
        """
        Tests whether each geometry in the GeoSeries is simple.

        Here "simple" means that a geometry has no anomalous point, such as a self-intersection or a self-tangency.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether an element is simple.

            * *True:* The geometry is simple.
            * *False:* The geometry is not simple.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT (1 1)", "LINESTRING (2 0, 2 2, 2 6)", "POLYGON ((3 3, 7 3, 7 7, 3 7, 3 3))"])
        >>> s.is_simple
        0    True
        1    True
        2    True
        Name: 0, dtype: bool
        """
        return _column_op("st_issimple", self)

    @property
    def geom_type(self):
        """
        Returns the type of each geometry in the GeoSeries.

        Returns
        -------
        Koalas Series
            The string representations of geometry types. For example, "ST_LINESTRING", "ST_POLYGON", "ST_POINT", and "ST_MULTIPOINT".

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT (1 1)", "LINESTRING (2 0, 2 2, 2 6)", "POLYGON ((3 3, 7 3, 7 7, 3 7, 3 3))"])
        >>> s.geom_type
        0         Point
        1    LineString
        2       Polygon
        Name: 0, dtype: object
        """
        return _column_op("st_geometrytype", self)

    @property
    def centroid(self):
        """
        Returns the centroid of each geometry in the GeoSeries.

        Returns
        -------
        GeoSeries
            The centroid of each geometry in the GeoSeries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT (1 1)", "LINESTRING (2 0, 2 2, 2 6)", "POLYGON ((3 3, 7 3, 7 7, 3 7, 3 3))"])
        >>> s.centroid
        0    POINT (1 1)
        1    POINT (2 3)
        2    POINT (5 5)
        Name: 0, dtype: object
        """
        return _column_geo("st_centroid", self, crs=self._crs)

    @property
    def convex_hull(self):
        """
        For each geometry in the GeoSeries, returns the smallest convex geometry that encloses it.

        * For a polygon, the returned geometry is the smallest convex geometry that encloses it.
        * For a geometry collection, the returned geometry is the smallest convex geometry that encloses all geometries in the collection.
        * For a point or line, the returned geometry is the same as the original.

        Returns
        -------
        GeoSeries
            Sequence of convex geometries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(2 0.5)", "LINESTRING(0 0,3 0.5)",  "POLYGON ((1 1,3 1,3 3,1 3, 1 1))"])
        >>> s.convex_hull
        0                          POINT (2 0.5)
        1                LINESTRING (0 0, 3 0.5)
        2    POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))
        Name: 0, dtype: object
        """
        return _column_geo("st_convexhull", self, crs=self._crs)

    @property
    def npoints(self):
        """
        Returns the number of points for each geometry in the GeoSeries.

        Returns
        -------
        Koalas Series
            Number of points of each geometry in the GeoSeries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(2 0.5)", "LINESTRING(0 0,3 0.5)",  "POLYGON ((1 1,3 1,3 3,1 3, 1 1))"])
        >>> s.npoints
        0    1
        1    2
        2    5
        Name: 0, dtype: int32
        """
        return _column_op("st_npoints", self)

    @property
    def envelope(self):
        """
        Returns the minimum bounding box for each geometry in the GeoSeries.

        The bounding box is a rectangular geometry object, and its edges are parallel to the axes.

        Returns
        -------
        GeoSeries
            Minimum bounding box of each geometry in the GeoSeries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT (1 1)", "LINESTRING (0 2, 2 2, 2 6)", "POLYGON ((3 3, 7 3, 7 7, 3 7, 3 3))"])
        >>> s.envelope
        0                            POINT (1 1)
        1    POLYGON ((0 2, 0 6, 2 6, 2 2, 0 2))
        2    POLYGON ((3 3, 3 7, 7 7, 7 3, 3 3))
        Name: 0, dtype: object
        """
        return _column_geo("st_envelope", self, crs=self._crs)

    @property
    def exterior(self):
        """
        For each geometry in the GeoSeries, returns a line string representing the exterior ring of the geometry.

        * For a polygon, the returned geometry is a line string representing its exterior ring.
        * For other geometries, returns None.

        Returns
        --------
        GeoSeries
            Sequence of line strings.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT (1 1)", "LINESTRING (0 2, 2 2, 2 6)", "POLYGON ((3 3, 7 3, 7 7, 3 7, 3 3))"])
        >>> s.exterior
        0                             POINT (1 1)
        1              LINESTRING (0 2, 2 2, 2 6)
        2    LINESTRING (3 3, 7 3, 7 7, 3 7, 3 3)
        Name: 0, dtype: object

        """
        return _column_geo("st_exteriorring", self)

    @property
    def is_empty(self):
        """
        Tests whether each geometry in the GeoSeries is empty.

        Returns
        --------
        Mask of boolean values for each element in the GeoSeries that indicates whether an element is empty.
            * *True:* The geometry is empty.
            * *False:* The geometry is not empty.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["LINESTRING EMPTY", "POINT (1 1)"])
        >>> s.is_empty
        0     True
        1    False
        Name: 0, dtype: bool
        """
        return _column_op("st_isempty", self)

    @property
    def boundary(self):
        """
        Returns the closure of the combinatorial boundary of each geometry in the GeoSeries.

        * For a polygon, the returned geometry is the same as the original.
        * For a geometry collection, the returned geometry is the combinatorial boundary of all geometries in the collection.
        * For a point or line, the returned geometry is an empty geometry collection.

        Returns
        -------
        GeoSeries
            The boundary (low-dimension) of each geometry in the GeoSeries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.boundary
        0                GEOMETRYCOLLECTION EMPTY
        1    LINESTRING (1 1, 3 1, 3 3, 1 3, 1 1)
        Name: 0, dtype: object
        """
        return _column_geo("st_boundary", self)

    # -------------------------------------------------------------------------
    # Geometry related unary methods, which return GeoSeries
    # -------------------------------------------------------------------------
    def make_valid(self):
        """
        Creates a valid representation of each geometry in the GeoSeries without losing any of the input vertices.

        If the geometry is already-valid, then nothing will be done. If the geometry can't be made to valid, it will be set to None.

        Returns
        -------
        GeoSeries
            Sequence of valid geometries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> import pandas as pd
        >>> pd.set_option("max_colwidth", 1000)
        >>> s = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 8,2 1))"])
        >>> s.make_valid()
        0    POLYGON ((2 1, 3 1, 3 2, 2 2, 2 8, 2 1))
        Name: 0, dtype: object
        """
        return _column_geo("st_makevalid", self, crs=self._crs)

    def precision_reduce(self, precision):
        """
        For the coordinates of each geometry in the GeoSeries, reduces the number of significant digits to the given number. The digit in the last decimal place will be rounded.

        Parameters
        ----------
        precision : int
            Number of significant digits.

        Returns
        -------
        GeoSeries
            Sequence of geometries with reduced precision.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT (1.3333 2.6666)", "POINT (2.6555 4.4447)"])
        >>> s.precision_reduce(3)
        0    POINT (1.333 2.667)
        1    POINT (2.656 4.445)
        Name: 0, dtype: object
        """
        return _column_geo("st_precisionreduce", self, F.lit(precision), crs=self._crs)

    def unary_union(self):
        """
        Calculates a geometry that represents the union of all geometries in the GeoSeries.

        Returns
        -------
        GeoSeries
            A GeoSeries that contains only one geometry, which is the union of all geometries in the GeoSeries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> p1 = "POINT(1 2)"
        >>> p2 = "POINT(1 1)"
        >>> s = GeoSeries([p1, p2])
        >>> s.unary_union()
        0    GEOMETRYCOLLECTION EMPTY
        Name: st_union_aggr(0), dtype: object
        """
        return _agg("st_union_aggr", self)

    def envelope_aggr(self):
        """
        Returns the minimum bounding box for the union of all geometries in the GeoSeries.

        The bounding box is a rectangular geometry object, and its sides are parallel to the axes.

        Returns
        -------
        GeoSeries
            A GeoSeries that contains only one geometry, which is the minimum bounding box for the union of all geometries in the GeoSeries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POLYGON ((0 0,4 0,4 4,0 4,0 0))", "POLYGON ((5 1,7 1,7 2,5 2,5 1))"])
        >>> s.envelope_aggr()
        0    POLYGON ((0 0, 0 4, 7 4, 7 0, 0 0))
        Name: ST_Envelope_Aggr(0), dtype: object
        """
        return _agg("st_envelope_aggr", self)

    def curve_to_line(self):
        """
        Converts curves in each geometry to approximate linear representations.

        For example,

        * CIRCULAR STRING to LINESTRING,
        * CURVEPOLYGON to POLYGON,
        * MULTISURFACE to MULTIPOLYGON.

        It is useful for outputting to devices that can't support CIRCULARSTRING geometry types.

        Returns
        -------
        GeoSeries
            Converted linear geometries.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))"])
        >>> rst = s.curve_to_line().to_wkt()
        >>> assert str(rst[0]).startswith("POLYGON")
        """
        return _column_geo("st_curvetoline", self, crs=self._crs)

    def simplify(self, tolerance):
        """
        Returns a simplified version for each geometry in the GeoSeries using the `Douglas-Peucker algorithm <https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm>`_.

        Parameters
        ----------
        tolerance : float
            Distance tolerance.

        Returns
        -------
        GeoSeries
            Sequence of simplified geometries.

        Examples
        .. doctest::
           :skipif: True
        -------
        >>> import matplotlib.pyplot as plt
        >>> from arctern_spark import GeoSeries
        >>> from arctern.plot import plot_geometry
        >>> g0 = GeoSeries(["CURVEPOLYGON(CIRCULARSTRING(0 0, 10 0, 10 10, 0 10, 0 0))"])
        >>> g0 = g0.curve_to_line()
        >>> fig, ax = plt.subplots()
        >>> ax.axis('equal') # doctest: +SKIP
        >>> ax.grid()
        >>> plot_geometry(ax,g0,facecolor="red",alpha=0.2)
        >>> plot_geometry(ax,g0.simplify(1),facecolor="green",alpha=0.2)
        """
        return _column_geo("st_simplifypreservetopology", self, F.lit(tolerance), crs=self._crs)

    def buffer(self, distance):
        """
        For each geometry, moves all of its points away from its centroid to construct a new geometry. The distance of movement is specified as ``distance``.

        * If ``distance`` > 0, the new geometry is a scaled-up version outside the original geometry.
        * If ``distance`` < 0, the new geometry is a scaled-down version inside the original geometry.

        Parameters
        ----------
        distance : float
            Distance of movement.

        Returns
        -------
        GeoSeries
            Sequence of geometries.

        Examples
        .. doctest::
           :skipif: True
        -------
        >>> import matplotlib.pyplot as plt
        >>> from arctern_spark import GeoSeries
        >>> from arctern.plot import plot_geometry
        >>> g0 = GeoSeries(["CURVEPOLYGON(CIRCULARSTRING(0 0, 10 0, 10 10, 0 10, 0 0))"])
        >>> g0 = g0.curve_to_line()
        >>> fig, ax = plt.subplots()
        >>> ax.axis('equal') # doctest: +SKIP
        >>> ax.grid()
        >>> plot_geometry(ax,g0,facecolor=["red"],alpha=0.2)
        >>> plot_geometry(ax,g0.buffer(-2),facecolor=["green"],alpha=0.2)
        >>> plot_geometry(ax,g0.buffer(2),facecolor=["blue"],alpha=0.2)
        """
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
        >>> from arctern_spark import GeoSeries
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

    def scale(self, factor_x, factor_y, origin="center"):
        """
        Scales the geometry to a new size by multiplying the ordinates with the corresponding factor parameters.

        Parameters
        ----------
        factor_x : float
            Scaling factor for x dimension.
        factor_y : float
            Scaling factor for y dimension.

        origin : string or tuple
            The point of origin can be a keyword ‘center’ for 2D bounding box center (default), ‘centroid’ for the geometry’s 2D centroid, or a coordinate tuple (x, y).

        Returns
        -------
        GeoSeries
            A GeoSeries that contains geometries with a new size by multiplying the ordinates with the corresponding factor parameters.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s1.scale(2,2)
        0    LINESTRING (-2.5 0, 7.5 0)
        1     MULTIPOINT ((3 0), (7 0))
        Name: 0, dtype: object
        >>> s1.scale(2, 2, (1, 1))
        0        LINESTRING (-1 -1, 9 -1)
        1    MULTIPOINT ((7 -1), (11 -1))
        Name: 0, dtype: object
        """
        if isinstance(origin, str):
            result = _column_geo("st_scale", self, F.lit(factor_x), F.lit(factor_y), F.lit(origin))
        elif isinstance(origin, tuple) and len(origin) == 2:
            result = _column_geo("st_scale", self, F.lit(factor_x), F.lit(factor_y), F.lit(origin[0]), F.lit(origin[1]))
        return result

    def affine(self, a, b, d, e, offset_x, offset_y):
        """
        Return a GeoSeries with transformed geometries.

        Parameters
        -----------
        a:
        b:
        d:
        e:
        offset_x:
        offset_y:

        Returns
        --------
        GeoSeries
            A GeoSeries that contains geometries which are tranformed by parameters in matrix.

        Examples
        ---------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s1.affine(2,2,2,2,2,2)
        0          LINESTRING (2 2, 12 12)
        1    MULTIPOINT ((10 10), (14 14))
        Name: 0, dtype: object
        """
        return _column_geo("st_affine", self, F.lit(a), F.lit(b), F.lit(d), F.lit(e), F.lit(offset_x), F.lit(offset_y))

    def translate(self, shifter_x, shifter_y):
        """
        Return a GeoSeries with translated geometries.

        Parameters
        ----------
        shifter_x : float
            Amount of offset along x dimension.
        shifter_y : float
            Amount of offset along y dimension.

        Returns
        -------
        GeoSeries
            A GeoSeries with translated geometries which shifted by offsets along each dimension.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s.translate(2, 1)
        0        LINESTRING (2 1, 7 1)
        1    MULTIPOINT ((6 1), (8 1))
        Name: 0, dtype: object
        """
        return _column_geo("st_translate", self, F.lit(shifter_x), F.lit(shifter_y))

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return Series[bool/float]
    # -------------------------------------------------------------------------

    def intersects(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether they intersect each other.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether it is intersected with the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function tests the intersection relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the intersection relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it intersects the geometries in ``other``.

            * *True*: The two geometries intersect each other.
            * *False*: The two geometries do not intersect each other.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.intersects(s1)
        0    True
        1    True
        Name: 0, dtype: bool

        Alternatively, ``other`` can be a geometry in WKB format.

        >>> s2.intersects(s1[0])
        0    True
        1    True
        Name: 0, dtype: bool
        """
        return _column_op("st_intersects", self, _validate_arg(other)).astype(bool)

    def within(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether the first geometry is within the other.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether the geometries in the first GeoSeries is within it.

            * If ``other`` is a geometry, this function tests the "within" relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the "within" relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it is within the geometries in ``other``.
            * *True*: The first geometry is within the other.
            * *False*: The first geometry is not within the other.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.within(s1)
        0    False
        1    False
        Name: 0, dtype: bool
        """
        return _column_op("st_within", self, _validate_arg(other)).astype(bool)

    def contains(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether the first geometry contains the other.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether the geometries in the first GeoSeries contains it.

            * If ``other`` is a geometry, this function tests the "contain" relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the "contain" relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it contains the geometries in ``other``.

            * *True*: The first geometry contains the other.
            * *False*: The first geometry does not contain the other.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.contains(s1)
        0     True
        1    False
        Name: 0, dtype: bool
        """
        return _column_op("st_contains", self, _validate_arg(other)).astype(bool)

    def geom_equals(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether the first geometry equals the other.

        "Equal" means two geometries represent the same geometry structure.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether it equals the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function tests the equivalence relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the equivalence relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it equals the geometries in ``other``.

            * *True*: The first geometry equals the other.
            * *False*: The first geometry does not equal the other.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.geom_equals(s1)
        0    False
        1    False
        Name: 0, dtype: bool
        """
        return _column_op("st_equals", self, _validate_arg(other)).astype(bool)

    def crosses(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether the first geometry spatially crosses the other.

        "Spatially cross" means two geometries have some, but not all interior points in common. The intersection of the interiors of the geometries must not be the empty set and must have a dimensionality less than the maximum dimension of the two input geometries.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether it crosses the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function tests the "cross" relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the "cross" relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it crosses the geometries in ``other``.

            * *True*: The first geometry crosses the other.
            * *False*: The first geometry does not cross the other.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.crosses(s1)
        0    False
        1    False
        Name: 0, dtype: bool
        """
        return _column_op("st_crosses", self, _validate_arg(other)).astype(bool)

    def touches(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether the first geometry touches the other.

        "Touch" means two geometries have common points, and the common points locate only on their boundaries.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether it touches the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function tests the "touch" relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the "touch" relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it touches the geometries in ``other``.

            * *True*: The first geometry touches the other.
            * *False*: The first geometry does not touch the other.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.touches(s1)
        0    False
        1     True
        Name: 0, dtype: bool
        """
        return _column_op("st_touches", self, _validate_arg(other)).astype(bool)

    def overlaps(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether the first geometry "spatially overlaps" the other.

        "Spatially overlap" here means two geometries intersect but one does not completely contain another.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether it overlaps the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function tests the "overlap" relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the "overlap" relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it overlaps the geometries in ``other``.

            * *True*: The first geometry overlaps the other.
            * *False*: The first geometry does not overlap the other.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.overlaps(s1)
        0    False
        1    False
        Name: 0, dtype: bool
        """
        return _column_op("st_overlaps", self, _validate_arg(other)).astype(bool)

    def distance(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, calculates the minimum 2D Cartesian (planar) distance between them.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to calculate the distance between it and the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function calculates the distance between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function calculates the distance between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Distance between each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> p11 = "LINESTRING(9 0,9 2)"
        >>> p12 = "POINT(10 2)"
        >>> s1 = GeoSeries([p11, p12])
        >>> p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
        >>> p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
        >>> s2 = GeoSeries([p21, p22])
        >>> s2.distance(s1)
        0    1.0
        1    2.0
        Name: 0, dtype: float64
        """
        return _column_op("st_distance", self, _validate_arg(other))

    def distance_sphere(self, other):
        """
        For each point in the GeoSeries and the corresponding point given in ``other``, calculates the minimum spherical distance between them.

        This function uses a spherical earth and radius derived from the spheroid defined by the SRID.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to calculate the spherical distance between it and the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function calculates the spherical distance between each geometry in the GeoSeries and ``other``. The ``crs`` of ``other`` is "EPSG:4326" bu default.
            * If ``other`` is a GeoSeries, this function calculates the spherical distance between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Spherical distance between each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Notes
        -------
        Only the longitude and latitude coordinate reference system ("EPSG:4326") can be used to calculate spherical distance.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POINT(10 2)"], crs="EPSG:4326")
        >>> s2 = GeoSeries(["POINT(10 3)"], crs="EPSG:4326")
        >>> s2.distance_sphere(s1)
        0    111226.3
        Name: 0, dtype: float64
        """
        return _column_op("st_distancesphere", self, _validate_arg(other))

    def hausdorff_distance(self, other):
        """
        For each point in the GeoSeries and the corresponding point given in ``other``, calculates the Hausdorff distance between them.

        Hausdorff distance is a measure of how similar two geometries are.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to calculate the Hausdorff distance between it and the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function calculates the Hausdorff distance between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function calculates the Hausdorff distance between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Hausdorff distance between each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0 ,0 1, 1 1, 1 0, 0 0))", "POINT(0 0)"])
        >>> s2 = GeoSeries(["POLYGON((0 0 ,0 2, 1 1, 1 0, 0 0))", "POINT(0 1)"])
        >>> s2.hausdorff_distance(s1)
        0    1.0
        1    1.0
        Name: 0, dtype: float64
        """
        return _column_op("st_hausdorffdistance", self, _validate_arg(other))

    def disjoint(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether they do not intersect each other.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether it is not intersected with the geometries in the first GeoSeries.
            * If ``other`` is a geometry, this function tests the intersection relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the intersection relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Koalas Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it intersects the geometries in ``other``.
            * *True*: The two geometries do not intersect each other.
            * *False*: The two geometries intersect each other.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.disjoint(s1)
        0    False
        1     True
        Name: 0, dtype: bool
        """
        return _column_op("st_disjoint", self, _validate_arg(other))

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def intersection(self, other):
        """
        For each point in the GeoSeries and the corresponding point given in ``other``, calculates the intersection of them.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to calculate the intersection of it and the geometries in the first GeoSeries.

            * If ``other`` is a geometry, this function calculates the intersection of each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function calculates the intersection of each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        GeoSeries
            Intersection of each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
        >>> s2 = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
        >>> s2.intersection(s1)
        0    LINESTRING (2 2, 2 1)
        Name: 0, dtype: object
        """
        return _column_geo("st_intersection", self, _validate_arg(other), crs=self.crs)

    def difference(self, other):
        """
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, returns a geometry representing the part of the first geometry that does not intersect with the other.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to calculate the difference from the first GeoSeries.
            * If ``other`` is a geometry, this function calculates the difference between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function calculates the difference between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        GeoSeries
            A GeoSeries that contains geometries representing the difference between each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s2 = GeoSeries(["LINESTRING (4 0,6 0)", "POINT (4 0)"])
        >>> s1.difference(s2)
        0    LINESTRING (0 0, 4 0)
        1              POINT (6 0)
        Name: 0, dtype: object
        """
        return _column_geo("st_difference", self, _validate_arg(other))

    def symmetric_difference(self, other):
        """
        Returns a geometry that represents the portions of self and other that do not intersect.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to calculate the the portions of self and other that do not intersect.
            * If ``other`` is a geometry, this function calculates the sym difference of each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function calculates the sym difference of each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        GeoSeries
            A GeoSeries that contains geometries that represents the portions of self and other that do not intersect.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s2 = GeoSeries(["LINESTRING (4 0,6 0)", "POINT (4 0)"])
        >>> s1.symmetric_difference(s2)
        0    MULTILINESTRING ((0 0, 4 0), (5 0, 6 0))
        1                                 POINT (6 0)
        Name: 0, dtype: object
        """
        return _column_geo("st_symdifference", self, _validate_arg(other))

    def union(self, other):
        """
        This function returns a geometry being a union of two input geometries
        Parameters
        ----------
        other : GeoSeries
            The GeoSeries to calculate the union of it and the geometries in the first GeoSeries.

        Returns
        -------
        GeoSeries
            A GeoSeries that is the union of each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0 ,0 1, 1 1, 1 0, 0 0))", "POINT(0 0)"])
        >>> s2 = GeoSeries(["POLYGON((0 0 ,0 2, 1 1, 1 0, 0 0))", "POINT(0 1)"])
        >>> s2.union(s1)
        0    POLYGON ((0 0, 0 1, 0 2, 1 1, 1 0, 0 0))
        1                   MULTIPOINT ((0 0), (0 1))
        Name: 0, dtype: object
        """
        return _column_geo("st_union", self, _validate_arg(other))

    # -------------------------------------------------------------------------
    # Geometry related quaternary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def rotate(self, rotation_angle, origin=None, use_radians=False):
        """
        Returns a rotated geometry on a 2D plane.
        Parameters
        ----------
        rotation_angle : float
            The angle of rotation which can be specified in either degrees (default) or radians by setting use_radians=True. Positive angles are counter-clockwise and negative are clockwise rotations.
        origin : string or tuple
            The point of origin can be a keyword ‘center’ for 2D bounding box center (default), ‘centroid’ for the geometry’s 2D centroid, or a coordinate tuple (x, y).
        use_radians : boolean
            Whether to interpret the angle of rotation as degrees or radians.

        Returns
        -------
        GeoSeries
            A GeoSeries with rotated geometries.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> import math
        >>> s1.rotate(90, (0,1)).precision_reduce(3)
        0        LINESTRING (1 1, 1 6)
        1    MULTIPOINT ((1 5), (1 7))
        Name: 0, dtype: object
        """
        import math
        result = None
        if not use_radians:
            rotation_angle = rotation_angle * math.pi / 180.0

        if origin is None:
            result = _column_geo("st_rotate", self, F.lit(rotation_angle))
        elif isinstance(origin, str):
            result = _column_geo("st_rotate", self, F.lit(rotation_angle), F.lit(origin))
        elif isinstance(origin, tuple):
            origin_x = origin[0]
            origin_y = origin[1]
            result = _column_geo("st_rotate", self, F.lit(rotation_angle), F.lit(origin_x), F.lit(origin_y))
        return result

    # -------------------------------------------------------------------------
    # utils
    # -------------------------------------------------------------------------

    @classmethod
    def polygon_from_envelope(cls, min_x, min_y, max_x, max_y, crs=None):
        """
        Constructs rectangular POLYGON objects within the given spatial range. The edges of the rectangles are parallel to the coordinate axises.

        ``min_x``, ``min_y``, ``max_x``, and ``max_y`` are Series so that polygons can be created in batch. The number of values in the four Series should be the same.

        Suppose that the demension of ``min_x`` is *N*, the returned GeoSeries of this function should contains *N* rectangles. The shape and position of the rectangle with index *i* is defined by its bottom left vertex *(min_x[i], min_y[i])* and top right vertex *(max_x[i], max_y[i])*.

        Parameters
        ----------
        min_x : Series
            The minimum x coordinates of the rectangles.
        min_y : Series
            The minimum y coordinates of the rectangles.
        max_x : Series
            The maximum x coordinates of the rectangles.
        max_y : Series
            The maximum y coordinates of the rectangles.
        crs : str, optional
            A string representation of Coordinate Reference System (CRS).
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, "EPSG:4326".

        Returns
        -------
        GeoSeries
            Sequence of rectangular POLYGON objects within the given spatial range.

        Examples
        -------
        >>> from pandas import Series
        >>> from arctern_spark import GeoSeries
        >>> min_x = Series([0.0, 1.0])
        >>> max_x = Series([2.0, 1.5])
        >>> min_y = Series([0.0, 1.0])
        >>> max_y = Series([1.0, 1.5])
        >>> GeoSeries.polygon_from_envelope(min_x, min_y, max_x, max_y)
        0            POLYGON ((0 0, 0 1, 2 1, 2 0, 0 0))
        1    POLYGON ((1 1, 1 1.5, 1.5 1.5, 1.5 1, 1 1))
        Name: min_x, dtype: object
        """
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
        """
        Constructs POINT objects based on the given coordinates.

        ``x`` and ``y`` are Series so that points can be created in batch. The number of values in the two Series should be the same.

        Suppose that the demension of ``x`` is *N*, the returned GeoSeries of this function should contains *N* points. The position of the *i*th point is defined by its coordinates *(x[i], y[i]).*

        Parameters
        ----------
        x : Series
            X coordinates of points.
        y : Series
            Y coordinates of points.
        crs : str, optional
            A string representation of Coordinate Reference System (CRS).
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, "EPSG:4326".

        Returns
        -------
        GeoSeries
            Sequence of POINT objects.

        Examples
        -------
        >>> from pandas import Series
        >>> from arctern_spark import GeoSeries
        >>> x = Series([1.3, 2.5])
        >>> y = Series([1.3, 2.5])
        >>> GeoSeries.point(x, y)
        0    POINT (1.3 1.3)
        1    POINT (2.5 2.5)
        Name: 0, dtype: object
        """
        dtype = (float, int)
        return _column_geo("st_point", *_validate_args(x, y, dtype=dtype), crs=crs)

    @classmethod
    def geom_from_geojson(cls, json, crs=None):
        """
        Constructs geometries from GeoJSON strings.

        ``json`` is Series so that geometries can be created in batch.

        Parameters
        ----------
        json : Series
            String representations of geometries in JSON format.
        crs : str, optional
            A string representation of Coordinate Reference System (CRS).
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, "EPSG:4326".

        Returns
        -------
        GeoSeries
            Sequence of geometries.

        Examples
        -------
        >>> from pandas import Series
        >>> from arctern_spark import GeoSeries
        >>> json = Series(['{"type":"LineString","coordinates":[[1,2],[4,5],[7,8]]}'])
        >>> GeoSeries.geom_from_geojson(json)
        0    LINESTRING (1 2, 4 5, 7 8)
        Name: 0, dtype: object
        """
        if not isinstance(json, (pd.Series, ks.Series)) and is_list_like(json) and not json:
            return GeoSeries([], crs=crs)
        return _column_geo("st_geomfromgeojson", _validate_arg(json), crs=crs)

    def as_geojson(self):
        """
        Transforms all geometries in the GeoSeries to GeoJSON strings.

        Returns
        -------
        Koalas Series
            Sequence of geometries in GeoJSON format.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)"])
        >>> s.as_geojson()
        0    {"type":"Point","coordinates":[1.0,1.0]}
        Name: 0, dtype: object
        """
        return _column_op("st_asgeojson", self)

    def to_wkt(self):
        """
        Transforms all geometries in the GeoSeries to `WKT <https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry>`_ strings.

        Returns
        -------
        Koalas Series
            Sequence of geometries in WKT format.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)"])
        >>> s.to_wkt()
        0    POINT (1 1)
        Name: 0, dtype: object
        """
        return _column_op("st_astext", self)

    def to_wkb(self):
        """
        Transforms all geometries in the GeoSeries to `WKB <https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry#Well-known_binary>`_ strings.

        Returns
        -------
        Koalas Series
            Sequence of geometries in WKB format.

        Examples
        -------
        >>> from arctern_spark import GeoSeries
        >>> import pandas as pd
        >>> pd.set_option("max_colwidth", 1000)
        >>> s = GeoSeries(["POINT(1 1)"])
        >>> s.to_wkb()
        0    [0, 0, 0, 0, 1, 63, 240, 0, 0, 0, 0, 0, 0, 63, 240, 0, 0, 0, 0, 0, 0]
        Name: 0, dtype: object
        """
        return _column_op("st_aswkb", self)

    def head(self, n: int = 5):
        """
        Return the first n rows.

        This function returns the first n rows for the object based on position.
        It is useful for quickly testing if your object has the right type of data in it.

        Parameters
        ----------
        n : Integer, default =  5

        Returns
        -------
        GeoSeries
            The first n rows of the GeoSeries.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))", "LINESTRING (0 0, 1 1, 1 7)", "POINT(4 4)"])
        >>> s.head(2)
        0    POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))
        1             LINESTRING (0 0, 1 1, 1 7)
        Name: 0, dtype: object
        """
        r = super().head(n)
        r.set_crs(self.crs)
        return r

    def take(self, indices):
        """
        Return the elements in the given *positional* indices along an axis.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.

        Returns
        -------
        GeoSeries
            An array-like containing the elements taken from the GeoSeries.

        Examples
        --------
        >>> from arctern_spark import GeoSeries
        >>> s = GeoSeries(["POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))", "LINESTRING (0 0, 1 1, 1 7)", "POINT(4 4)"])
        >>> s.take([0,2])
        0    POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))
        2                            POINT (4 4)
        Name: 0, dtype: object
        """
        r = super().take(indices)
        r.set_crs(self.crs)
        return r

    @classmethod
    def _calculate_bbox_from_wkb(cls, geom_wkb):
        """
        Calculate bounding box for the geom_wkb geometry.
        """
        from osgeo import ogr
        geometry = ogr.CreateGeometryFromWkb(geom_wkb)
        env = geometry.GetEnvelope()
        return [env[0], env[2], env[1], env[3]]

    @property
    def bbox(self):
        """
        Calculate bounding box for the each geometry in the GeoSeries.

        :rtype: a Pandas Series with each item is a (minx, miny, maxx, maxy) list
        :return: Bounding box of each geometry.
        """
        envelope = self.envelope.to_pandas().apply(GeoSeries._calculate_bbox_from_wkb)
        return envelope

    def iterfeatures(self, na="null", show_bbox=False):
        """
        Returns an iterator that yields feature dictionaries that comply with
        Arctern.GeoSeries.

        Parameters
        ----------
        na: str {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame
            * null: ouput the missing entries as JSON null
            * drop: remove the property from the feature. This applies to
                    each feature individually so that features may have
                    different properties
            * keep: output the missing entries as NaN

        show_bbox: bool
            whether to include bbox (bounds box) in the geojson. default False
        """
        import json
        if na not in ["null", "drop", "keep"]:
            raise ValueError("Unknown na method {0}".format(na))

        ids = self.to_pandas()
        for fid, geom in zip(ids, self.to_pandas()):
            feature = {
                "id": str(fid),
                "type": "Feature",
                "properties": {},
                "geometry": json.loads(GeoSeries(geom).as_geojson()[0]) if geom else None,
            }
            if show_bbox:
                feature["bbox"] = GeoSeries._calculate_bbox_from_wkb(geom) if geom else None
            yield feature

    @classmethod
    def from_file(cls, fp, bbox=None, mask=None, item=None, **kwargs):
        """
        Read a file or OGR dataset to GeoSeries.

        Supported file format is listed in
        https://github.com/Toblerity/Fiona/blob/master/fiona/drvsupport.py.

        Parameters
        ----------
        fp: URI (str or pathlib.Path), or file-like object
            A dataset resource identifier or file object.

        bbox: a (minx, miny, maxx, maxy) tuple
            Filter for geometries which spatial intersects with by the provided bounding box.

        mask: a GeoSeries(should have same crs), wkb formed bytes or wkt formed string
            Filter for geometries which spatial intersects with by the provided geometry.

        item: int or slice
            Load special items by skipping over items or stopping at a specific item.

        **kwargs: Keyword arguments to `fiona.open()`. e.g. `layer`, `enabled_drivers`.
                       see https://fiona.readthedocs.io/en/latest/fiona.html#fiona.open for
                       more info.

        Returns
        ----------
            A GeoSeries read from file.
        """
        import fiona
        import json
        with fiona.Env():
            with fiona.open(fp, "r", **kwargs) as features:
                if features.crs is not None:
                    crs = features.crs.get("init", None)
                else:
                    crs = features.crs_wkt

                if mask is not None:
                    if isinstance(mask, (str, bytes)):
                        mask = GeoSeries(mask)
                    if not isinstance(mask, GeoSeries):
                        raise TypeError(f"unsupported mask type {type(mask)}")
                    mask = mask.unary_union().as_geojson()
                if isinstance(item, (int, type(None))):
                    item = (item,)
                elif isinstance(item, slice):
                    item = (item.start, item.stop, item.step)
                else:
                    raise TypeError(f"unsupported item type {type(item)}")
                features = features.filter(*item, bbox=bbox, mask=mask)

                geoms = []
                for feature in features:
                    geometry = feature["geometry"]
                    geoms.append(json.dumps(geometry) if geometry is not None else '{"type": "null"}')
                return GeoSeries.geom_from_geojson(geoms, crs=crs)

    def to_file(self, fp, mode="w", driver="ESRI Shapefile", **kwargs):
        """
        Store GeoSeries to a file or OGR dataset.

        :type fp: URI (str or pathlib.Path), or file-like object
        :param fp: A dataset resource identifier or file object.

        :type mode: str, default "w"
        :param mode: 'a' to append, or 'w' to write. Not all driver support
                      append, see "Supported driver list" below for more info.

        :type driver: str, default "ESRI Shapefile"
        :param driver: The OGR format driver. It's  represents a
                       translator for a specific format. Supported driver is listed in
                       https://github.com/Toblerity/Fiona/blob/master/fiona/drvsupport.py.

        :param kwargs: Keyword arguments to `fiona.open()`. e.g. `layer` used to
                       write data to multi-layer dataset.
                       see https://fiona.readthedocs.io/en/latest/fiona.html#fiona.open for
                       more info.
        """

        geo_types = "Unknown"
        if len(self.geom_type) != 0:
            geo_types = set(self.geom_type.dropna().unique().to_pandas())

        schema = {"properties": {}, "geometry": geo_types}
        # TODO: fiona expected crs like Proj4 style mappings, "EPSG:4326" or WKT representations
        crs = self.crs
        import fiona
        with fiona.Env():
            with fiona.open(fp, mode, driver, crs=crs, schema=schema, **kwargs) as sink:
                sink.writerecords(self.iterfeatures())


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
