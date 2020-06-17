import arctern
import databricks.koalas as ks
import pandas as pd
from databricks.koalas import DataFrame, get_option, Series
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


def _validate_arg(arg, dtype):
    if isinstance(arg, dtype):
        arg = F.lit(arg)
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
        return _column_geo("ST_Centroid", self, crs=self._crs)

    @property
    def convex_hull(self):
        return _column_geo("ST_ConvexHull", self, crs=self._crs)

    @property
    def npoints(self):
        return _column_op("ST_NPoints", self)

    @property
    def envelope(self):
        return _column_geo("ST_Envelope", self, crs=self._crs)

    # -------------------------------------------------------------------------
    # Geometry related unary methods, which return GeoSeries
    # -------------------------------------------------------------------------
    def make_valid(self):
        return _column_geo("ST_MakeValid", self, crs=self._crs)

    def precision_reduce(self, precision):
        return _column_geo("ST_PrecisionReduce", self, F.lit(precision), crs=self._crs)

    def unary_union(self):
        return _column_geo("ST_Union_Aggr", self, crs=self._crs)

    def envelope_aggr(self):
        return _column_geo("ST_Envelope_Aggr", self, crs=self._crs)

    def curve_to_line(self):
        return _column_geo("ST_CurveToLine", self, crs=self._crs)

    def simplify(self, tolerance):
        return _column_geo("ST_SimplifyPreserveTopology", self, F.lit(tolerance), crs=self._crs)

    def buffer(self, buffer):
        return _column_geo("ST_Buffer", self, F.lit(buffer), crs=self._crs)

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
        return _column_geo("ST_Transform", self, F.lit(self.crs), F.lit(crs), crs=crs)

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return Series[bool/float]
    # -------------------------------------------------------------------------

    def intersects(self, other):
        return _column_op("ST_Intersects", self, _validate_arg(other, dtype=bytearray))

    def within(self, other):
        return _column_op("ST_Within", self, _validate_arg(other, dtype=bytearray))

    def contains(self, other):
        return _column_op("ST_Contains", self, _validate_arg(other, dtype=bytearray))

    def geom_equals(self, other):
        return _column_op("ST_Equals", self, _validate_arg(other, dtype=bytearray))

    def crosses(self, other):
        return _column_op("ST_Crosses", self, _validate_arg(other, dtype=bytearray))

    def touches(self, other):
        return _column_op("ST_Touches", self, _validate_arg(other, dtype=bytearray))

    def overlaps(self, other):
        return _column_op("ST_Overlaps", self, _validate_arg(other, dtype=bytearray))

    def distance(self, other):
        return _column_op("ST_Distance", self, _validate_arg(other, dtype=bytearray))

    def distance_sphere(self, other):
        return _column_op("ST_DistanceSphere", self, _validate_arg(other, dtype=bytearray))

    def hausdorff_distance(self, other):
        return _column_op("ST_HausdorffDistance", self, _validate_arg(other, dtype=bytearray))

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def intersection(self, other):
        return _column_geo("ST_Intersection", self, _validate_arg(other, dtype=bytearray), crs=self.crs)

    @classmethod
    def polygon_from_envelope(cls, min_x, min_y, max_x, max_y, crs=None):
        dtype = (float, int)
        # min_x, min_y, max_x, max_y = _validate_args(min_x, min_y, max_x, max_y, dtype=dtype)
        arg_list = _validate_args(min_x, min_y, max_x, max_y, dtype=dtype)
        for arg in arg_list:
            print(type(arg))
        return _column_geo("ST_PolygonFromEnvelope", *arg_list, crs=crs)

    @classmethod
    def point(cls, x, y, crs=None):
        dtype = (float, int)
        return _column_geo("ST_Point", *_validate_args(x, y, dtype=dtype), crs=crs)

    @classmethod
    def geom_from_geojson(cls, json, crs=None):
        return _column_geo("ST_GeomFromGeoJSON", Series(json), crs=crs)

    def to_wkt(self):
        return _column_op("ST_AsText", self)


if __name__ == "__main__":
    from pandas import Series as pds
    x_min = pds([0.0])
    x_max = pds([1.0])
    y_min = pds([0.0])
    y_max = pds([1.0])

    # rst = GeoSeries.polygon_from_envelope(x_min, y_min, x_max, y_max).to_wkt()

    # assert rst[0] == "POLYGON ((0 0,0 1,1 1,1 0,0 0))"


    def test_ST_Point():
        from databricks.koalas import Series
        data1 = [1.3, 2.5]
        data2 = [3.8, 4.9]
        string_ptr = GeoSeries.point(data1, data2).to_wkt()
        assert len(string_ptr) == 2
        assert string_ptr[0] == "POINT (1.3 3.8)"
        assert string_ptr[1] == "POINT (2.5 4.9)"

        string_ptr = GeoSeries.point(Series([1, 2], dtype='double'), 5).to_wkt()
        assert len(string_ptr) == 2
        assert string_ptr[0] == "POINT (1 5)"
        assert string_ptr[1] == "POINT (2 5)"

        string_ptr = GeoSeries.point(5, Series([1, 2], dtype='double')).to_wkt()
        assert len(string_ptr) == 2
        assert string_ptr[0] == "POINT (5 1)"
        assert string_ptr[1] == "POINT (5 2)"

        string_ptr = GeoSeries.point(5.0, 1.0).to_wkt()
        assert len(string_ptr) == 1
        assert string_ptr[0] == "POINT (5 1)"
    test_ST_Point()