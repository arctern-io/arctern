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

import databricks.koalas as ks
import pandas as pd
from databricks.koalas import DataFrame, Series, get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.series import first_series, REPR_PATTERN
from pandas.io.formats.printing import pprint_thing
from pyspark.sql import functions as F

# os.environ['PYSPARK_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"
# os.environ['PYSPARK_DRIVER_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"

ks.set_option('compute.ops_on_diff_frames', True)


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
    from . import scala_wrapper
    scol = getattr(scala_wrapper, f)(kss.spark_column)
    sdf = kss._internal._sdf.select(scol)
    kdf = sdf.to_koalas()
    return GeoSeries(first_series(kdf), crs=kss._crs)


def _validate_crs(crs):
    if crs is not None and not isinstance(crs, str):
        raise TypeError("`crs` should be spatial reference identifier string")
    crs = crs.upper() if crs is not None else crs
    return crs


def _validate_arg(arg, dtype):
    if isinstance(arg, dtype):
        arg = F.lit(arg)
    elif not isinstance(arg, Series):
        arg = Series(arg)
    return arg


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
                assert not copy
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
            kss = first_series(anchor)
            column_label = anchor._internal.column_labels[0]

            from pyspark.sql.types import StringType, BinaryType
            from .scala_wrapper import GeometryUDT
            spark_dtype = kss.spark.data_type
            if isinstance(spark_dtype, GeometryUDT):
                pass
            if isinstance(spark_dtype, BinaryType):
                pass
            elif isinstance(spark_dtype, StringType):
                kss = _column_op("st_geomfromtext", kss)
            else:
                raise TypeError("Can not use no StringType or BinaryType or GeometryUDT data to construct GeoSeries.")
            anchor = kss.to_dataframe()
            anchor._kseries = {column_label: kss}

        super(Series, self).__init__(anchor)
        self._column_label = column_label
        self.set_crs(crs)

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
        return _column_geo("ST_CurveToLine", self, crs=self._crs)

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

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return Series[bool/float]
    # -------------------------------------------------------------------------

    def intersects(self, other):
        return _column_op("st_intersects", self, _validate_arg(other, bytearray))

    def within(self, other):
        return _column_op("st_within", self, _validate_arg(other, bytearray))

    def contains(self, other):
        return _column_op("st_contains", self, _validate_arg(other, bytearray))

    def geom_equals(self, other):
        return _column_op("st_equals", self, _validate_arg(other, bytearray))

    def crosses(self, other):
        return _column_op("st_crosses", self, _validate_arg(other, bytearray))

    def touches(self, other):
        return _column_op("st_touches", self, _validate_arg(other, bytearray))

    def overlaps(self, other):
        return _column_op("st_overlaps", self, _validate_arg(other, bytearray))

    def distance(self, other):
        return _column_op("st_distance", self, _validate_arg(other, bytearray))

    def distance_sphere(self, other):
        return _column_op("st_distancesphere", self, _validate_arg(other, bytearray))

    def hausdorff_distance(self, other):
        return _column_op("st_hausdorffdistance", self, _validate_arg(other, bytearray))

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def intersection(self, other):
        return _column_geo("st_intersection", self, _validate_arg(other, bytearray), crs=self.crs)

    @classmethod
    def polygon_from_envelope(cls, min_x, min_y, max_x, max_y, crs=None):
        dtype = (float, int)
        return _column_geo("st_polygonfromenvelope", _validate_arg(min_x, dtype), _validate_arg(min_y, dtype),
                           _validate_arg(max_x, dtype), _validate_arg(max_y, dtype), crs=crs)

    @classmethod
    def point(cls, x, y, crs=None):
        dtype = (float, int)
        return _column_geo("st_point", _validate_arg(x, dtype), _validate_arg(y, dtype), crs=crs)

    @classmethod
    def geom_from_geojson(cls, json, crs=None):
        return _column_geo("st_geomfromgeojson", Series(json), crs=crs)

    def to_wkt(self):
        return _column_op("st_astext", self)
