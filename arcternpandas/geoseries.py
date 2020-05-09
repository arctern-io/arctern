from pandas import Series
from arcternpandas.geoarray import GeoArray, GeoDtype, is_geometry_arry
import arctern
from warnings import warn


def _property_op(op, this):
    # type: (function, GeoSeries) -> Series[bool/float]
    return Series(op(this.values).values, index=this.index)


def _unary_op(op, this, *args, **kwargs):
    # type: (function, GeoSeries, args/kwargs) -> GeoSeries
    crs = kwargs.pop('crs')
    if crs is None:
        crs = this.crs
    return GeoSeries(op(this.values, *args, **kwargs).values, index=this.index, name=this.name, crs=crs)


def _delegate_binary_op(op, this, other):
    # type: (function, GeoSeries, GeoSeries) -> GeoSeries/Series
    if isinstance(other, GeoSeries):
        if not this.index.equals(other.index):
            warn("The indices of the two GeoSeries are different.")
            this, other = this.align(other)
    else:
        raise TypeError(type(this), type(other))
    data = op(this.values, other.values)
    return data, this.index


def _binary_op(op, this, other):
    # type: (function, GeoSeries, GeoSeries) -> Series[bool/float]
    # TODO: support other is single geometry
    data, index = _delegate_binary_op(op, this, other)
    return Series(data.values, index=index)


def _binary_geo(op, this, other):
    # type: (function, GeoSeries, GeoSeries) -> GeoSeries
    data, index = _delegate_binary_op(op, this, other)
    return GeoSeries(data.values, index=index, crs=this.crs)


class GeoSeries(Series):
    _metadata = ["name"]

    def __init__(self, data=None, index=None, dtype=None, name=None, crs=None, **kwargs):
        if hasattr(data, "crs") and crs:
            if not data.crs:
                data = data.copy()
            else:
                raise ValueError(
                    "CRS mismatch between CRS of the passed geometries and crs."
                )
        if isinstance(data, bytes):
            n = len(index) if index is not None else 1
            data = [data] * n

        if not is_geometry_arry(data):
            s = Series(data, index=index, name=name, **kwargs)
            if not s.dtype == object:
                if s.empty:
                    s = s.astype(bytes)
                else:
                    raise TypeError("Can not use no bytes data to construct GeoSeries.")
            data = GeoArray(s.values, crs=crs)

        super().__init__(data, index=index, dtype=dtype, name=name, **kwargs)

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
        return _property_op(arctern.ST_Centroid, self)

    @property
    def convex_hull(self):
        return _property_op(arctern.ST_ConvexHull, self)

    @property
    def npoints(self):
        return _property_op(arctern.ST_NPoints, self)

    @property
    def envelope(self):
        return _property_op(arctern.ST_Envelope, self)

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
            raise ValueError("Can not transform native geometries. Set crs for this GeoSeries first.")
        # crs = CRS.from_user_input(crs)
        # if self.crs.is_exact_same(crs):
        #     return self
        if crs == self.crs:
            return self
        return _unary_op(arctern.ST_Transform, self, crs=crs, src=self.crs, dst=crs)

    def simplify_preserve_to_pology(self, distance_tolerance):
        return _unary_op(arctern.ST_SimplifyPreserveTopology, self, distance_tolerance=distance_tolerance)

    def projection(self, bottom_right, top_left, height, width):
        return _unary_op(arctern.projection, self)

    def transform_and_projection(self, src_rs, dst_rs, bottom_right, top_left, height, width):
        return _unary_op(arctern.transform_and_projection, self, src_rs, dst_rs, bottom_right, top_left, height, width)

    def buffer(self, distance):
        return _unary_op(arctern.ST_Buffer, self, distance)

    def precision_reduce(self, precision):
        return _unary_op(arctern.ST_PrecisionReduce, self, precision)

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
