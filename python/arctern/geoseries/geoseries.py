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

# pylint: disable=useless-super-delegation
# pylint: disable=too-many-lines
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-ancestors, protected-access

from warnings import warn
from pandas import Series, DataFrame
import arctern
from .geoarray import GeoArray, is_geometry_array, GeoDtype


def fix_dataframe_box_col_volues():
    def _box_col_values(self, values, items):
        klass = self._constructor_sliced

        if isinstance(values.dtype, GeoDtype):
            klass = GeoSeries

        return klass(values, index=self.index, name=items, fastpath=True)

    DataFrame._box_col_values = _box_col_values


fix_dataframe_box_col_volues()


def _property_op(op, this):
    # type: (function, GeoSeries) -> Series[bool/float/object]
    return Series(op(this).values, index=this.index)


def _property_geo(op, this):
    # type: (function, GeoSeries) -> GeoSeries
    return GeoSeries(op(this).values, index=this.index, crs=this.crs)


def _unary_geo(op, this, *args, **kwargs):
    # type: (function, GeoSeries, args, kwargs) -> GeoSeries
    crs = kwargs.pop("crs", this.crs)
    return GeoSeries(op(this, *args, **kwargs).values, index=this.index, name=this.name, crs=crs)


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
    data = op(this, other).values
    return data, this.index


def _binary_op(op, this, other):
    # type: (function, GeoSeries, GeoSeries/bytes) -> Series[bool/float]
    data, index = _delegate_binary_op(op, this, other)
    return Series(data, index=index)


def _binary_geo(op, this, other):
    # type: (function, GeoSeries, GeoSeries/bytes) -> GeoSeries
    data, index = _delegate_binary_op(op, this, other)
    return GeoSeries(data, index=index, crs=this.crs)


def _validate_crs(crs):
    if crs is not None and not isinstance(crs, str):
        raise TypeError("`crs` should be spatial reference identifier string")
    crs = crs.upper() if crs is not None else crs
    return crs


class GeoSeries(Series):
    """
    A Series to store geometry data which is WKB formed bytes object.

    :type data: array-like, Iterable, dict, or scalar value(str or bytes)
    :param data: Geometries to store, which can be WKB formed bytes or WKT formed string.

    :type index: array-like or Index (1d)
    :param index: Same to Pandas Series.
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If both a dict and index
        sequence are used, the index will override the keys found in the
        dict.

    :type name: str, optional
    :param name: The name to give to the Series.

    :type crs: str, optional
    :param crs: The coordinate system for the GeoSeries, now only support SRID form.

    :param kwargs: Additional arguments passed to the GeoSeries constructor, e.g. ``copy``

    :example:
    >>> from arctern import GeoSeries
    >>> s = GeoSeries(["POINT(1 1)", "POINT(1 2)"])
    >>> s
    0    POINT (1 1)
    1    POINT (1 2)
    dtype: GeoDtype
    """

    _metadata = ["name"]

    def __init__(self, data=None, index=None, name=None, crs=None, **kwargs):

        if hasattr(data, "crs") and crs:
            if not data.crs:
                data = data.copy()
            elif not data.crs == crs:
                raise ValueError(
                    "csr of the passed geometry data is different from crs."
                )
        # scalar wkb or wkt
        if isinstance(data, (bytes, str)):
            n = len(index) if index is not None else 1
            data = [data] * n

        if not is_geometry_array(data):
            s = Series(data, index=index, name=name, **kwargs)
            index = s.index
            name = s.name
            if s.empty:
                s = s.astype(object)
            # make sure missing value is None
            s[s.isna()] = None
            from pandas.api.types import infer_dtype
            inferred = infer_dtype(s, skipna=True)
            if inferred in ("bytes", "empty"):
                pass
            elif inferred == "string":
                s = arctern.ST_GeomFromText(s)
            else:
                raise TypeError("Can not use no bytes or string data to construct GeoSeries.")
            data = GeoArray(s.values)

        super().__init__(data, index=index, name=name, **kwargs)

        self._crs = None
        self.set_crs(crs)

    def set_crs(self, crs):
        """
        Set the coordinate system for the GeoSeries.

        :type crs: str, optional
        :param crs: SRID(spatial reference identifier) form.
        """
        crs = _validate_crs(crs)
        self._crs = crs

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, crs):
        """
        Set the coordinate system for the GeoSeries.

        :type crs: str, optional
        :param crs: SRID(spatial reference identifier) form.
        """
        self.set_crs(crs)

    @property
    def _constructor(self):
        # Some operations result is not geometry type, we should return Series as constructor
        # e.g.(isna, notna)
        def _try_constructor(data, index=None, crs=self.crs, **kwargs):
            try:
                from pandas.core.internals import SingleBlockManager
                # astype will dispatch to here,Only if `dtype` is `GeoDtype`
                # will return GeoSeries
                if isinstance(data, SingleBlockManager):
                    dtype = getattr(data, 'dtype')
                    if not isinstance(dtype, GeoDtype):
                        raise TypeError
                return GeoSeries(data, index=index, crs=crs, **kwargs)
            except TypeError:
                return Series(data, index=index, **kwargs)

        return _try_constructor

    # --------------------------------------------------------------------------
    # override Series method
    # --------------------------------------------------------------------------

    def equals(self, other):
        """
        Test whether two objects contain the same elements.

        This function allows two GeoSeries to be compared against each other to
        see if they have the same shape and geometries (same wkb bytes).
        NaNs in the same location are considered equal. The column headers do not
        need to have the same type, but the elements within the columns must
        be the same dtype.

        :type other: GeoSeries
        :param other: The other GeoSeries to be compared with the first

        :rtype: bool
        :return: True if all geometries are the same in both objects, False otherwise.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POINT(1 1)", None])
        >>> s2 = GeoSeries(["POINT(1 1)", None])
        >>> s2.equals(s1)
        True
        """
        if not isinstance(other, GeoSeries):
            return False
        return self._data.equals(other._data)

    def fillna(
            self,
            value=None,
            method=None,
            axis=None,
            inplace=False,
            limit=None,
            downcast=None,
    ):
        """
        Fill NA values with a geometry, which can be WKT or WKB formed.
        """
        return super().fillna(value, method, axis, inplace, limit, downcast)

    def isna(self):
        """
        Detect missing values.

        NA value in GeoSeries is represented as None.

        :rtype: Series(dtype: bool)
        :return: Mask of bool values for each element in GeoSeries
                that indicates whether an element is not an NA value.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["Point (1 1)", None])
        >>> s.isna()
        0    False
        1     True
        dtype: bool
        """
        return super().isna()

    def notna(self):
        """
        Detect non-missing values.

        Inverse of isna.

        :rtype: Series(dtype: bool)
        :return: Mask of bool values for each element in GeoSeries
                that indicates whether an element is not an NA value.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT (1 1)", None])
        >>> s.isna()
        0    False
        1     True
        dtype: bool
        """
        return super().notna()

    # -------------------------------------------------------------------------
    # Geometry related property
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        """
        Check if each geometry is of valid geometry format.

        :rtype: Series(dtype: bool)
        :return: True value for geometries that are valid, False otherwise.

        :examples:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT(1)"])
        >>> s.is_valid
        0     True
        1    False
        dtype: bool
        """
        return _property_op(arctern.ST_IsValid, self).astype(bool, copy=False)

    @property
    def length(self):
        """
        Calculate the length of each geometry.

        :rtype: Series(dtype: float64)
        :return: The length of each geometry in the GeoSeries.

        :examples:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "LINESTRING (0 0, 0 2)"])
        >>> s.length
        0    0.0
        1    2.0
        dtype: float64
        """
        return _property_op(arctern.ST_Length, self)

    @property
    def is_simple(self):
        """
        Check whether each geometry is "simple".

        "Simple" here means that a geometry has no anomalous geometric points,
        such as self intersection or self tangency.

        :rtype: Series(dtype: bool)
        :return: True for geometries that are simple, False otherwise.

        :examples:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT EMPTY"])
        >>> s.is_simple
        0     True
        1    False
        dtype: bool
        """
        return _property_op(arctern.ST_IsSimple, self).astype(bool, copy=False)

    @property
    def area(self):
        """
        Calculate the 2D Cartesian (planar) area of each geometry.

        :rtype: Series(dtype: float64)
        :return: The area of each geometry.

        :examples:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.area
        0    0.0
        1    4.0
        dtype: float64
        """
        return _property_op(arctern.ST_Area, self)

    @property
    def geom_type(self):
        """
        For each geometry in geometries, return a string that indicates is type.

        :rtype: Series(dtype: object)
        :return: The type of geometry, e.g. "ST_LINESTRING", "ST_POLYGON", "ST_POINT", "ST_MULTIPOINT"

        :examples:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.geometry_type
        0      ST_POINT
        1    ST_POLYGON
        dtype: object
        """
        return _property_op(arctern.ST_GeometryType, self)

    @property
    def centroid(self):
        """
        Compute the centroid of each geometry.

        :rtype: GeoSeries
        :return: The centroid of geometries.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.centroid
        0    POINT (1 1)
        1    POINT (2 2)
        dtype: GeoDtype
        """
        return _property_geo(arctern.ST_Centroid, self)

    @property
    def convex_hull(self):
        """
        For each geometry, compute the smallest convex geometry
        that encloses all geometries in it.

        :rtype: GeoSeries
        :return: Convex Geometries.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.convex_hull
        0                        POINT (1 1)
        1    POLYGON ((1 1,1 3,3 3,3 1,1 1))
        dtype: GeoDtype
        """
        return _property_geo(arctern.ST_ConvexHull, self)

    @property
    def npoints(self):
        """
        Calculates the points number for each geometry.

        :rtype: Series(dtype: int)
        :return: The number of points for each geometry.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.npoints
        0    1
        1    5
        dtype: int64
        """
        return _property_op(arctern.ST_NPoints, self)

    @property
    def envelope(self):
        """
        Compute the double-precision minimum bounding box geometry for each geometry.

        :rtype: GeoSeries
        :return: bounding box geometries

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.envelope
        0                        POINT (1 1)
        1    POLYGON ((1 1,1 3,3 3,3 1,1 1))
        dtype: GeoDtype
        """
        return _property_geo(arctern.ST_Envelope, self)

    # -------------------------------------------------------------------------
    # Geometry related unary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def curve_to_line(self):
        """
        Convert curves in each geometry to approximate linear representation,
        e.g., CIRCULAR STRING to regular LINESTRING, CURVEPOLYGON to POLYGON,
        and MULTISURFACE to MULTIPOLYGON. Useful for outputting to devices
        that can't support CIRCULARSTRING geometry types.

        :rtype: GeoSeries
        :return: Converted geometries

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))"])
        >>> rst = s.curve_to_line().to_wkt()
        >>> assert str(rst[0]).startswith("POLYGON")
        """
        return _unary_geo(arctern.ST_CurveToLine, self)

    def to_crs(self, crs):
        """
        Transform each geometry to a different coordinate reference system.
        The ``crs`` attribute on the current GeoSeries must be set.

        :type crs: string
        :param crs: Coordinate Reference System of the geometry objects.
                    Must be SRID formed, e.g. "EPSG:4326"

        :rtype: GeoSeries
        :return: Geometries with transformed coordinate reference system.

        :example:
        >>> from arctern import GeoSeries
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
            raise ValueError("Can not transform geometries without crs. Set crs for this GeoSeries first.")
        if self.crs == crs:
            return self
        return _unary_geo(arctern.ST_Transform, self, self.crs, crs, crs=crs)

    def simplify(self, tolerance):
        """
        Returns a "simplified" version for each geometry using the Douglas-Peucker algorithm.

        :type: tolerance: float
        :param tolerance: The maximum distance between a point on a linestring and a curve.

        :rtype: GeoSeries
        :return: Simplified geometries.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "CIRCULARSTRING (0 0,1 1,2 0)"])
        >>> s.simplify(1)
        0    POLYGON ((1 1,1 2,2 2,2 1,1 1))
        1               LINESTRING (0 0,2 0)
        dtype: GeoDtype
        """
        return _unary_geo(arctern.ST_SimplifyPreserveTopology, self, tolerance)

    def buffer(self, distance):
        """
        For each geometry, returns a geometry that represents all points
        whose distance from this geos is less than or equal to "distance".

        :type distance: float
        :param distance: he maximum distance of the returned geometry from each geometry.

        :rtype: GeoSeries
        :return: Geometries.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT (0 1)"])
        >>> s.buffer(0)
        0    POLYGON EMPTY
        dtype: GeoDtype
        """
        return _unary_geo(arctern.ST_Buffer, self, distance)

    def precision_reduce(self, precision):
        """
        For the coordinates of each geometry, reduce the number of significant digits
        to the given number. The last decimal place will be rounded.

        :type precision: int
        :param precision: The number of significant digits.

        :rtype: GeoSeries
        :return: Geometries with reduced precision.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT (1.333 2.666)", "POINT (2.655 4.447)"])
        >>> s.precision_reduce(3)
        0    POINT (1.33 2.67)
        1    POINT (2.66 4.45)
        dtype: GeoDtype
        """
        return _unary_geo(arctern.ST_PrecisionReduce, self, precision)

    def make_valid(self):
        """
        Create a valid representation of each geometry without losing any of the input vertices.

        If the geometry is already-valid, then nothing will be done. If the geometry can't be
        made to valid, it will be set to None value.

        :rtype: GeoSeries
        :return: Geometries that are made to valid.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 8,2 1))"])
        >>> s.make_valid()
        0    GEOMETRYCOLLECTION (POLYGON ((2 2,3 2,3 1,2 1,2 2)),LINESTRING (2 2,2 8))
        dtype: GeoDtype
        """
        return _unary_geo(arctern.ST_MakeValid, self)

    def unary_union(self):
        """
        Return a geometry that represents the union of all geometries in the GeoSeries.

        :rtype: GeoSeries
        :return: A GeoSeries contains only one geometry.

        :example:
        >>> from arctern import GeoSeries
        >>> p1 = "POINT(1 2)"
        >>> p2 = "POINT(1 1)"
        >>> s = GeoSeries([p1, p2])
        >>> s.union_aggr()
        0    MULTIPOINT (1 2,1 1)
        dtype: GeoDtype
        """
        return GeoSeries(arctern.ST_Union_Aggr(self))

    def envelope_aggr(self):
        """
        Compute the double-precision minimum bounding box geometry for the union of all geometries.

        :rtype: GeoSeries
        :return: A GeoSeries contains only one geometry.

        :example:
        >>> from arctern import GeoSeries
        >>> p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
        >>> p2 = "POLYGON ((5 1,7 1,7 2,5 2,5 1))"
        >>> s = GeoSeries([p1, p2])
        >>> s.envelope_aggr()
        0    POLYGON ((0 0,0 4,7 4,7 0,0 0))
        dtype: GeoDtype
        """
        return GeoSeries(arctern.ST_Envelope_Aggr(self))

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return Series[bool/float]
    # -------------------------------------------------------------------------
    def intersects(self, other):
        """
        Check whether each geometry intersects other (elementwise).

        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to test if is intersected.
                      Can be scalar WKB formed bytes object, or a GeoSeries.

        :rtype : Series(dtype: bool)
        :return: A Series with value True if intersected.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.intersects(s1)
        0    True
        1    True
        dtype: bool

        Alternatively other can be ca scalar bytes object.

        >>> s2.intersects(s1[0])
        0    True
        1    True
        dtype: bool
        """
        return _binary_op(arctern.ST_Intersects, self, other).astype(bool, copy=False)

    def within(self, other):
        """
        Check whether each geometry is within other (elementwise).


        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to test if each geometry is within.
                      Can be scalar WKB formed bytes object, or a GeoSeries.

        :rtype: Series(dtype: bool)
        :return: A Series with value True if each geometry is within other.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.within(s1)
        0    False
        1    False
        dtype: bool
        """
        return _binary_op(arctern.ST_Within, self, other).astype(bool, copy=False)

    def contains(self, other):
        """
        Check whether each geometry contains other (elementwise).

        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to test if each geometry is contained.
                      Can be scalar WKB formed bytes object, or a GeoSeries.

        :rtype: Series(dtype: bool)
        :return: A Series with value True if each geometry contains other.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.contains(s1)
        0     True
        1    False
        dtype: bool
        """
        return _binary_op(arctern.ST_Contains, self, other).astype(bool, copy=False)

    def crosses(self, other):
        """
        Check whether each geometry and other(elementwise) "spatially cross".

        "Spatially cross" here means two geometries have
        some, but not all interior points in common. The intersection of the
        interiors of the geometries must not be the empty set and must have
        a dimensionality less than the maximum dimension of the two input geometries.

        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to test if cross.
                      Can be scalar WKB formed bytes object, or a GeoSeries.

        :rtype: Series(dtype: bool)
        :return: A Series with value True if each geometry crosses other.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.crosses(s1)
        0    False
        1    False
        dtype: bool
        """
        return _binary_op(arctern.ST_Crosses, self, other).astype(bool, copy=False)

    def geom_equals(self, other):
        """
        Check whether each geometry is "spatially equal" to other.

        "Spatially equal" means two geometries represent the same geometry structure.

        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to test if each geometry is equals.
                      Can be scalar WKB formed bytes object, or a GeoSeries.

        :rtype: Series(dtype: bool)
        :return: A Series with value True if each geometry is spatially equals to other.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.geom_equals(s1)
        0    False
        1    False
        dtype: bool
        """
        from pandas.api.types import is_scalar
        if is_scalar(other):
            other = self.__class__([other] * len(self), index=self.index)
        this = self
        if not this.index.equals(other.index):
            warn("The indices of the two GeoSeries are different.")
            this, other = this.align(other)
        result = _binary_op(arctern.ST_Equals, this, other).astype(bool, copy=False)
        other_na = other.isna()
        result[other_na & this.isna()] = True
        return result

    def touches(self, other):
        """
        Check whether each geometry "touches" other.

        "Touch" means two geometries have common points, and the
        common points locate only on their boundaries.

        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to test if each geometry is touched.
                      Can be scalar WKB formed bytes object, or a GeoSeries.

        :rtype: Series(dtype: bool)
        :return: A Series with value True if each geometry touches other.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.touches(s1)
        0    False
        1     True
        dtype: bool
        """
        return _binary_op(arctern.ST_Touches, self, other).astype(bool, copy=False)

    def overlaps(self, other):
        """
        Check whether each geometry "spatially overlaps" other.

        "Spatially overlap" here means two geometries intersect
        but one does not completely contain another.

        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to test if each geometry overlaps.
                      Can be scalar WKB formed bytes object, or a GeoSeries.

        :rtype: Series(dtype: bool)
        :return: A Series with value True if each geometry overlaps other.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.overlaps(s1)
        0    False
        1    False
        dtype: bool
        """
        return _binary_op(arctern.ST_Overlaps, self, other).astype(bool, copy=False)

    def distance(self, other):
        """
        Calculates the minimum 2D Cartesian (planar) distance between each geometry and other.

        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to calculate the distance to each geometry.
                      Can be scalar WKB formed bytes object, or a GeoSeries.

        :rtype: Series(dtype: float64)
        :return: A Series contains the distances between each geometry and other.

        :example:
        >>> from arctern import GeoSeries
        >>> p11 = "LINESTRING(9 0,9 2)"
        >>> p12 = "POINT(10 2)"
        >>> s1 = GeoSeries([p11, p12])
        >>> p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
        >>> p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
        >>> s2 = GeoSeries([p21, p22])
        >>> s2.distance(s1)
        0    1.0
        1    2.0
        dtype: float64
        """
        return _binary_op(arctern.ST_Distance, self, other)

    def distance_sphere(self, other):
        """
        Return minimum distance in meters between two lon/lat points.

        Uses a spherical earth and radius derived from the spheroid defined by the SRID.
        Only 'EPSG:4326' can calculate spherical distance.

        :type other: scalar bytes object geometry or GeoSeries
        :param other: The geometries to calculate the spherical distance to each geometry.
                      Can be scalar WKB formed bytes object, or a GeoSeries.
                      If other is scalar bytes object, we will assume other's crs is 'EPSG:4326'.

        :rtype: Series(dtype: float64)
        :return: A Series contains the spherical distance between each geometry and other.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POINT(10 2)"], crs="EPSG:4326")
        >>> s2 = GeoSeries(["POINT(10 3)"], crs="EPSG:4326")
        >>> s2.distance_sphere(s1)
        0    111226.3
        dtype: float64
        """
        if not self.crs == getattr(other, "crs", "EPSG:4326") == "EPSG:4326":
            raise ValueError("Only can calculate spherical distance with 'EPSG:4326' crs.")
        return _binary_op(arctern.ST_DistanceSphere, self, other)

    def hausdorff_distance(self, other):
        """
        Returns the Hausdorff distance between each geometry and other.

        This is a measure of how similar or dissimilar 2 geometries are.

        :type other: scalar bytes object geometry or GeoSeries
                     Can be scalar WKB formed bytes object, or a GeoSeries.
        :param other: The geometries to calculate the hausdorff distance to each geometry.

        :rtype: Series(dtype: float64)
        :return : A Series contains the hausdorff distance between each geometry and other.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0 ,0 1, 1 1, 1 0, 0 0))", "POINT(0 0)"])
        >>> s2 = GeoSeries(["POLYGON((0 0 ,0 2, 1 1, 1 0, 0 0))", "POINT(0 1)"])
        >>> s2.hausdorff_distance(s1)
        0    1.0
        1    1.0
        dtype: float64
        """
        return _binary_op(arctern.ST_HausdorffDistance, self, other)

    # -------------------------------------------------------------------------
    # Geometry related binary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def intersection(self, other):
        """
        Calculate the point set intersection between each geometry and other.

        :type other: scalar bytes object geometry or GeoSeries.
                      Can be scalar WKB formed bytes object, or a GeoSeries.
        :param other: The geometries to calculate the intersection point set between each geometry.

        :rtype: GeoSeries
        :return: A GeoSeries contains geometries.

        :example:
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
        >>> s2 = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
        >>> s2.intersection(s1)
        0    LINESTRING (2 2,2 1)
        dtype: GeoDtype
        """
        return _binary_geo(arctern.ST_Intersection, self, other)

    # -------------------------------------------------------------------------
    # utils
    # -------------------------------------------------------------------------

    def to_wkt(self):
        """
        Transform each geometry to WKT formed string.

        :rtype: Series(dtype: object)
        :return: A Series contains geometries as WKT formed string.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)"])
        >>> s
        0    POINT (1 1)
        dtype: GeoDtype
        >>> s.to_wkt()
        0    POINT (1 1)
        dtype: object
        """
        return _property_op(arctern.ST_AsText, self)

    def to_wkb(self):
        """
        Transform each geometry to WKB formed bytes object.

        :rtype: Series(dtype: object)
        :return: A Series contains geometries as WKB formed bytes object.
        """
        return _property_op(lambda x: x, self)

    def as_geojson(self):
        """
        Transform each to GeoJSON format string.

        :rtype: Series(dtype: object)
        :return: A Series contains geometries as GeoJSON formed string.

        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)"])
        >>> s
        0    POINT (1 1)
        dtype: GeoDtype
        >>> s.as_geojson()
        0    { "type": "Point", "coordinates": [ 1.0, 1.0 ] }
        dtype: object
        """
        return _property_op(arctern.ST_AsGeoJSON, self)

    def to_geopandas(self):
        """
        Transform each arctern GeoSeries to GeoPandas GeoSeries.

        :rtype: GeoPandas GeoSeries(dtype: geometry)
        :return: A GeoPandas GeoSeries.
        :example:
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)"])
        >>> s
        0    POINT (1 1)
        dtype: GeoDtype
        >>> s.to_geopandas()
        0    POINT (1.00000 1.00000)
        dtype: geometry
        """
        import geopandas
        import shapely

        return geopandas.GeoSeries(self.apply(lambda x: shapely.wkb.loads(x) if x is not None else None), crs=self.crs)

    @classmethod
    def polygon_from_envelope(cls, min_x, min_y, max_x, max_y, crs=None):
        """
        Construct polygon(rectangle) geometries from arr_min_x, arr_min_y, arr_max_x,
        arr_max_y and special coordinate system. The edges of polygon are parallel to coordinate axis.

        :type min_x: Series(dtype: float64)
        :param min_x: The minimum value of x coordinate of the rectangles.

        :type min_y: Series(dtype: float64)
        :param min_y: The minimum value of y coordinate of the rectangles.

        :type max_x: Series(dtype: float64)
        :param max_x: The maximum value of x coordinate of the rectangles.

        :type max_y: Series(dtype: float64)
        :param max_y: The maximum value of y coordinate of the rectangles.

        :type crs: string, optional
        :param crs: Must be SRID format string.

        :rtype: GeoSeries
        :return: A GeoSeries contains geometries.

        :example:
        >>> from pandas import Series
        >>> from arctern import GeoSeries
        >>> min_x = Series([0.0, 1.0])
        >>> max_x = Series([2.0, 1.5])
        >>> min_y = Series([0.0, 1.0])
        >>> max_y = Series([1.0, 1.5])
        >>> GeoSeries.polygon_from_envelope(min_x, min_y, max_x, max_y)
        0                POLYGON ((0 0,0 1,2 1,2 0,0 0))
        1    POLYGON ((1 1,1.0 1.5,1.5 1.5,1.5 1.0,1 1))
        dtype: GeoDtype
        """
        crs = _validate_crs(crs)
        return cls(arctern.ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y), crs=crs)

    @classmethod
    def point(cls, x, y, crs=None):
        """
        Construct Point geometries according to the coordinates.

        :type x: Series(dtype: float64)
        :param x: Abscissa of the point.

        :type y: Series(dtype: float64)
        :param y: Ordinate of the point.

        :type crs: string, optional
        :param crs: Must be SRID format string.

        :rtype: GeoSeries
        :return: A GeoSeries contains point geometries.

        :example:
        >>> from pandas import Series
        >>> from arctern import GeoSeries
        >>> x = Series([1.3, 2.5])
        >>> y = Series([1.3, 2.5])
        >>> GeoSeries.point(x, y)
        0    POINT (1.3 1.3)
        1    POINT (2.5 2.5)
        dtype: GeoDtype
        """
        crs = _validate_crs(crs)
        return cls(arctern.ST_Point(x, y), crs=crs)

    @classmethod
    def geom_from_geojson(cls, json, crs=None):
        """
        Construct geometry from the GeoJSON representation string.

        :type json: Series(dtype: object)
        :param json: Geometries in json format.

        :type crs: string, optional
        :param crs: Must be SRID format string.

        :rtype: GeoSeries
        :return: A GeoSeries contains geometries.

        :example:
        >>> from pandas import Series
        >>> from arctern import GeoSeries
        >>> json = Series(["{\"type\":\"LineString\",\"coordinates\":[[1,2],[4,5],[7,8]]}"])
        >>> GeoSeries.geom_from_geojson(json)
        0    LINESTRING (1 2,4 5,7 8)
        dtype: GeoDtype
        """
        crs = _validate_crs(crs)
        return cls(arctern.ST_GeomFromGeoJSON(json), crs=crs)

    @classmethod
    def from_geopandas(cls, data):
        """
        Construct geometries from geopandas GeoSeries.

        :rtype data: geopandas.GeoSeries
        :param data: Source geometries data.

        :rtype: arctern.GeoSeries
        :return: A arctern.GeoSeries constructed from geopandas.GeoSeries.
        """

        import geopandas as gpd
        import shapely.wkb
        if not isinstance(data, gpd.GeoSeries):
            raise TypeError(f"data must be {gpd.GeoSeries}, got {type(data)}")

        if data.crs is not None:
            crs = data.crs.to_authority() or data.crs.source_crs.to_authority()
            crs = crs[0] + ':' + crs[1]
        else:
            crs = None

        def f(x):
            if x is None:
                return x
            return shapely.wkb.dumps(x)

        return cls(data.apply(f), crs=crs)
