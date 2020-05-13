from pandas import Series
from arcternpandas.geoarray import GeoArray, GeoDtype, is_geometry_arry
import arctern
from warnings import warn
import numpy as np


def _property_op(op, this):
    # type: (function, GeoSeries) -> Series[bool/float/object]
    return Series(op(this.values), index=this.index)


def _property_geo(op, this):
    # type: (function, GeoSeries) -> GeoSeries
    return GeoSeries(op(this.values), index=this.index, crs=this.crs)


def _unary_op(op, this, *args, **kwargs):
    # type: (function, GeoSeries, args, kwargs) -> GeoSeries
    crs = None
    if 'crs' in kwargs.keys():
        crs = kwargs.pop('crs')
    if crs is None:
        crs = this.crs
    return GeoSeries(op(this.values, *args, **kwargs), index=this.index, name=this.name, crs=crs)


def _delegate_binary_op(op, this, other):
    # type: (function, GeoSeries, GeoSeries/bytes) -> GeoSeries/Series
    if isinstance(other, GeoSeries):
        if not this.index.equals(other.index):
            warn("The indices of the two GeoSeries are different.")
            this, other = this.align(other)
    elif isinstance(other, bytes):
        pass
    else:
        raise TypeError(type(this), type(other))
    data = op(this.values, other)
    return data, this.index


def _binary_op(op, this, other):
    # type: (function, GeoSeries, GeoSeries/bytes) -> Series[bool/float]
    # TODO: support other is single geometry
    data, index = _delegate_binary_op(op, this, other)
    return Series(data, index=index)


def _binary_geo(op, this, other):
    # type: (function, GeoSeries, GeoSeries/bytes) -> GeoSeries
    data, index = _delegate_binary_op(op, this, other)
    return GeoSeries(data, index=index, crs=this.crs)


class GeoSeries(Series):
    _metadata = ["name"]

    def __init__(self, data=None, index=None, dtype=None, name=None, crs=None, **kwargs):
        if dtype is not None and not isinstance(dtype, GeoDtype):
            raise ValueError("'dtype' must be GeoDtype or None.")

        if hasattr(data, "crs") and crs:
            if not data.crs:
                data = data.copy()
            else:
                raise ValueError(
                    "CRS mismatch between CRS of the passed geometries and crs."
                )
        # scalar wkb or wkt
        if isinstance(data, (bytes, str)):
            n = len(index) if index is not None else 1
            data = [data] * n

        if not is_geometry_arry(data):
            s = Series(data, index=index, name=name, **kwargs)
            # find first valid data type
            first_valid = None
            for item in s:
                if item is not None or item is not np.nan:
                    first_valid = item
                    break

            if isinstance(first_valid, bytes):
                pass
            elif isinstance(first_valid, str):
                s = arctern.ST_GeomFromText(s)
            else:
                if s.empty:
                    s = s.astype(object)
                else:
                    raise TypeError("Can not use no bytes or string data to construct GeoSeries.")
            data = GeoArray(s.values, crs=crs)

        super().__init__(data, index=index, name=name, **kwargs)

        self.crs = crs

    def set_crs(self, crs):
        # crs = CRS.from_user_input(crs)
        self.crs = crs

    @property
    def isna(self):
        return super().isna()

    # -------------------------------------------------------------------------
    # Geometry related property
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        return _property_op(arctern.ST_IsValid, self)

    @property
    def length(self):
        return _property_op(arctern.ST_Length, self)

    @property
    def is_simple(self):
        return _property_op(arctern.ST_IsSimple, self)

    @property
    def area(self):
        return _property_op(arctern.ST_Area, self)

    @property
    def geometry_type(self):
        return _property_op(arctern.ST_GeometryType, self)

    @property
    def centroid(self):
        return _property_geo(arctern.ST_Centroid, self)

    @property
    def convex_hull(self):
        return _property_geo(arctern.ST_ConvexHull, self)

    @property
    def npoints(self):
        return _property_op(arctern.ST_NPoints, self)

    @property
    def envelope(self):
        return _property_geo(arctern.ST_Envelope, self)

    # -------------------------------------------------------------------------
    # Geometry related unary methods
    # -------------------------------------------------------------------------

    def curve_to_line(self):
        return _unary_op(arctern.ST_CurveToLine, self)

    def to_crs(self, crs):
        # TODO: should we support pyproj CRS?
        """
        Returns a new ``GeoSeries`` with all geometries transformed to a different spatial
        reference system. The ``crs`` attribute on the current GeoSeries must be set.

        :param crs: string.
                Coordinate Reference System of the geometry objects. such as authority string(eg "EPSG:4326")
        :return: GeoSeries
        """
        if crs is None:
            raise ValueError("Can not transform with invalid crs")
        if self.crs is None:
            raise ValueError("Can not transform geometries without crs. Set crs for this GeoSeries first.")
        # crs = CRS.from_user_input(crs)
        # if self.crs.is_exact_same(crs):
        #     return self
        if crs == self.crs:
            return self
        return _unary_op(arctern.ST_Transform, self, self.crs, crs, crs=crs)

    def simplify_preserve_to_pology(self, distance_tolerance):
        return _unary_op(arctern.ST_SimplifyPreserveTopology, self, distance_tolerance)

    def projection(self, bottom_right, top_left, height, width):
        return _unary_op(arctern.projection, self)

    def transform_and_projection(self, src_rs, dst_rs, bottom_right, top_left, height, width):
        return _unary_op(arctern.transform_and_projection, self, src_rs, dst_rs, bottom_right, top_left, height, width)

    def buffer(self, distance):
        return _unary_op(arctern.ST_Buffer, self, distance)

    def precision_reduce(self, precision):
        return _unary_op(arctern.ST_PrecisionReduce, self, precision)

    def make_valid(self):
        return _unary_op(arctern.ST_MakeValid, self)

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return Series[bool/float]
    # -------------------------------------------------------------------------
    def intersects(self, other):
        return _binary_op(arctern.ST_Intersects, self, other)

    def within(self, other):
        return _binary_op(arctern.ST_Within, self, other)

    def contains(self, other):
        return _binary_op(arctern.ST_Contains, self, other)

    def crosses(self, other):
        return _binary_op(arctern.ST_Crosses, self, other)

    def st_equals(self, other):
        return _binary_op(arctern.ST_Equals, self, other)

    def touches(self, other):
        return _binary_op(arctern.ST_Touches, self, other)

    def overlaps(self, other):
        return _binary_op(arctern.ST_Overlaps, self, other)

    def distance(self, other):
        return _binary_op(arctern.ST_Distance, self, other)

    def distance_sphere(self, other):
        return _binary_op(arctern.ST_DistanceSphere, self, other)

    def hausdorff_distance(self, other):
        return _binary_op(arctern.ST_HausdorffDistance, self, other)

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def intersection(self, other):
        return _binary_geo(arctern.ST_Intersection, self, other)

    # -------------------------------------------------------------------------
    # utils
    # -------------------------------------------------------------------------

    def to_wkt(self):
        return _property_op(arctern.ST_AsText, self)

    def to_wkb(self):
        return _property_op(lambda x: x, self)

    def as_geojson(self):
        return _property_op(arctern.ST_AsGeoJSON, self)
