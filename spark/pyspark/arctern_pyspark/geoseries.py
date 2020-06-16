import arctern
import databricks.koalas as ks
import pandas as pd
from databricks.koalas import DataFrame, get_option
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.internal import _InternalFrame
from databricks.koalas.series import REPR_PATTERN
from pandas.io.formats.printing import pprint_thing
from pyspark.sql import functions as F

# os.environ['PYSPARK_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"
# os.environ['PYSPARK_DRIVER_PYTHON'] = "/home/shengjh/miniconda3/envs/koalas/bin/python"

ks.set_option('compute.ops_on_diff_frames', True)


# for unary or binary operation, which return koalas Series.
def _column_op(f, *args):
    from arctern_pyspark import _wrapper_func
    return ks.base._column_op(getattr(_wrapper_func, f))(*args)


# for unary or binary operation, which return GeoSeries.
def _column_geo(f, *args, **kwargs):
    from arctern_pyspark import _wrapper_func
    kss = ks.base._column_op(getattr(_wrapper_func, f))(*args)
    return GeoSeries(kss._internal, anchor=kss._kdf, **kwargs)


def _validate_crs(crs):
    if crs is not None and not isinstance(crs, str):
        raise TypeError("`crs` should be spatial reference identifier string")
    crs = crs.upper() if crs is not None else crs
    return crs


class GeoSeries(ks.Series):
    def __init__(
            self, data=None, index=None, dtype=None, name=None, copy=False, crs=None, fastpath=False, anchor=None
    ):
        if isinstance(data, _InternalFrame):
            assert dtype is None
            assert name is None
            assert not copy
            assert not fastpath
            IndexOpsMixin.__init__(self, data, anchor)
        else:
            assert anchor is None
            if isinstance(data, arctern.GeoSeries):
                assert index is None
                assert dtype is None
                assert name is None
                assert not copy
                assert not fastpath
                if not data.crs == crs:
                    raise ValueError("csr of the passed geometry data is different from crs.")
                self.set_crs(crs)
                pds = data.astype(object, copy=False)
            else:
                pds = arctern.GeoSeries(
                    data=data, index=index, dtype=dtype, name=name, crs=crs, copy=copy, fastpath=fastpath
                )
                self.set_crs(pds.crs)
                pds = pds.astype(object, copy=False)
            kdf = DataFrame(pds)
            IndexOpsMixin.__init__(
                self, kdf._internal.copy(spark_column=kdf._internal.data_spark_columns[0]), kdf
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

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            pser = pd.Series(arctern.ST_AsText(self._to_internal_pandas()).to_numpy(),
                             name=self.name,
                             index=self.index.to_numpy(),
                             copy=False
                             )
            return pser.to_string(name=self.name, dtype=self.dtype)

        pser = pd.Series(arctern.ST_AsText(self.head(max_display_count + 1)._to_internal_pandas()).to_numpy(),
                         name=self.name,
                         index=self.index.to_numpy(),
                         copy=False
                         )
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
        return _column_op("ST_Area", self)

    @property
    def is_valid(self):
        return _column_op("ST_IsValid", self)

    @property
    def length(self):
        return _column_op("ST_Length", self)

    @property
    def is_simple(self):
        return _column_op("ST_IsSimple", self)

    @property
    def geom_type(self):
        return _column_op("ST_GeometryType", self)

    @property
    def centroid(self):
        return _column_geo("ST_Centroid", self)

    @property
    def convex_hull(self):
        return _column_geo("ST_ConvexHull", self)

    @property
    def npoints(self):
        return _column_op("ST_NPoints", self)

    @property
    def envelope(self):
        return _column_geo("ST_Envelope", self)

    # -------------------------------------------------------------------------
    # Geometry related unary methods, which return GeoSeries
    # -------------------------------------------------------------------------
    def make_valid(self):
        return _column_geo("ST_MakeValid", self)

    def precision_reduce(self, precision):
        return _column_geo("ST_PrecisionReduce", self, F.lit(precision))

    def unary_union(self):
        return _column_geo("ST_Union_Aggr", self)

    def envelope_aggr(self):
        return _column_geo("ST_Envelope_Aggr", self)

    def curve_to_line(self):
        return _column_geo("ST_CurveToLine", self)

    def simplify(self, tolerance):
        return _column_geo("ST_SimplifyPreserveTopology", self, tolerance)

    def buffer(self, buffer):
        return _column_geo("ST_Buffer", self, F.lit(buffer))

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
        return _column_geo("ST_Transform", self.crs, crs, crs=crs)

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return Series[bool/float]
    # -------------------------------------------------------------------------

    def intersects(self, other):
        return _column_op("ST_Intersects", self, other)

    def within(self, other):
        return _column_op("ST_Within", self, other)

    def contains(self, other):
        return _column_op("ST_Contains", self, other)

    def geom_equals(self, other):
        return _column_op("ST_Equals", self, other)

    def crosses(self, other):
        return _column_op("ST_Crosses", self, other)

    def touches(self, other):
        return _column_op("ST_Touches", self, other)

    def overlaps(self, other):
        return _column_op("ST_Overlaps", self, other)

    def distance(self, other):
        return _column_op("ST_Distance", self, other)

    def distance_sphere(self, other):
        return _column_op("ST_DistanceSphere", self, other)

    def hausdorff_distance(self, other):
        return _column_op("ST_HausdorffDistance", self, other)

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def intersection(self, other):
        return _column_geo("ST_Intersection", self, other, crs=self.crs)

    @classmethod
    def polygon_from_envelope(cls, min_x, min_y, max_x, max_y, crs=None):
        return _column_geo("ST_PolygonFromEnvelope", min_x, min_y, max_x, max_y, crs=crs)

    @classmethod
    def point(cls, x, y, crs=None):
        return _column_geo("ST_Point", x, y, crs=crs)

    def geom_from_geojson(self, json, crs=None):
        return _column_geo("ST_GeoFromGeoJson", json, crs=crs)
