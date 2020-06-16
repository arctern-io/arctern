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
import numpy as np
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
    return Series(np.asarray(op(this)), index=this.index)


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
    data = np.asarray(op(this, other))
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
    One-dimensional Series to store an array of geometry objects.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value(str or bytes)
        Contains geometric data stored in GeoSeries. The geometric data can be in `WKT <https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry>`_ or `WKB <https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry#Well-known_binary>`_ format.
    index : array-like or Index (1d)
        Same to the index of pandas.Series, by default ``RangeIndex (0, 1, 2, …, n)``.
        Index values must be hashable and have the same length as ``data``. Non-unique index values are allowed. If both a dict and index sequence are used, the index will override the keys found in the dict.
    name : str, optional
        The name to give to the GeoSeries.
    crs : str, optional
        The Coordinate Reference System (CRS) set to all geometries in GeoSeries.
        Only supports SRID as a WKT representation of CRS by now, for example, ``"EPSG:4326"``.
    **kwargs :
        Parameters to pass to the GeoSeries constructor, for example, ``copy``.
        ``copy`` is a boolean value, by default False.
        * *True:* Copys input data.
        * *False:* Points to input data.

    Examples
    -------
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

            from pandas.api.types import is_object_dtype, is_float_dtype
            # The default dtype for empty Series is 'float64' in pandas
            if not is_geometry_array(s):
                if is_float_dtype(s.dtype) and s.empty:
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
        Sets the Coordinate Reference System (CRS) for all geometries in GeoSeries.

        Parameters
        ----------
        crs : str
            A string representation of CRS.
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, ``"EPSG:4326"``.

        Notes
        -------
        Arctern supports common CRSs listed at the `Spatial Reference <https://spatialreference.org/>`_ website.

        Examples
        -------
        >>> from arctern import GeoSeries
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
        >>> from arctern import GeoSeries
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
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, ``"EPSG:4326"``.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POINT(1 2)"])
        >>> s.set_crs("EPSG:4326")
        >>> s.crs
        'EPSG:4326'
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
        Tests whether two objects contain the same elements.

        This function allows two GeoSeries to be compared against each other to see if they have the same shape and elements. NaNs in the same location are considered equal. The column headers do not need to have the same type, but the elements within the columns must be the same dtype.

        Parameters
        ----------
        other : GeoSeries
            The other GeoSeries to be compared with the first GeoSeries.

        Returns
        -------
        bool
            * *True:* All geometries are the same in both objects.
            * *False:* The two objects are different in shape or elements.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POINT(1 1)", None])
        >>> s2 = GeoSeries(["Point(1 1)", None])
        >>> s2.equals(s1)
        True

        >>> from arctern import GeoSeries
        >>> s3 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1))"])
        >>> s4 = GeoSeries(["POLYGON ((1 1,1 2,2 2,2 1))"])
        >>> s4.equals(s3)
        True

        >>> from arctern import GeoSeries
        >>> s5 = GeoSeries(["POLYGON ((0 0,2 0,2 2,0 2,0 0))"])
        >>> s6 = GeoSeries(["POLYGON ((0 0,0 2,2 2,2 0,0 0))"])
        >>> s6.equals(s5)
        False
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
        Fills NA values with a geometry in WKT or WKB format.

        Parameters
        ----------
        value :
            Value to use to fill holes. This value should be the representation of a geometry in WKT or WKB format. For exmaple, "POINT (1 1)".
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, by default None
            Method to use for filling holes in reindexed Series.
            * *pad/ffill:* Propagates the last valid observation forward to fill gap.
            * *backfill/bfill:* Propagates the next valid observation backward to fill gap.
        axis : {0 or 'index'}
            Axis along which to fill missing values.
        inplace : bool, by default False

            * *True:* Fills NA values in-place.
                **Note:** This will modify any other views on this object.
            * *False:* Create a new GeoSeries object, and then fill NA values in it.

        limit : int, by default None
            * If ``method`` is specified, this is the maximum number of consecutive
            NA values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NAs, it will only
            be partially filled.
            * If ``method`` is not specified, this is the maximum number of entries along the entire axis where NaNs will be filled.
            **Note:** It must be greater than 0 if not None.
        downcast : dict, by default None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (for example, float64 to int64 if possible).

        Returns
        -------
        GeoSeries or None
            * None if ``inplace=False``.
            * A GeoSeries with missing values filled if ``inplace=True``.

        Examples
        -------
        >>> s = GeoSeries(["Point (1 1)", "POLYGON ((1 1,2 1,2 2,1 2,1 1))", None, ""])
        >>> s
        0                        POINT (1 1)
        1    POLYGON ((1 1,2 1,2 2,1 2,1 1))
        2                               None
        3                               None
        dtype: GeoDtype

        Replace all NA elements with a string in WKT format.

        >>> s.fillna("POINT (1 2)")
        0                        POINT (1 1)
        1    POLYGON ((1 1,2 1,2 2,1 2,1 1))
        2                        POINT (1 2)
        3                        POINT (1 2)
        dtype: GeoDtype

        We can also replace all NA elements with an object in WKB format. Here we use ``s[1]`` as an example of WKB objects because ``s`` is a GeoSeries and all elements stored in GeoSeries are transformed to WKB format.

        >>> s.fillna(s[1])
        0                        POINT (1 1)
        1    POLYGON ((1 1,2 1,2 2,1 2,1 1))
        2    POLYGON ((1 1,2 1,2 2,1 2,1 1))
        3    POLYGON ((1 1,2 1,2 2,1 2,1 1))
        dtype: GeoDtype
        """
        return super().fillna(value, method, axis, inplace, limit, downcast)

    def isna(self):
        """
        Detects missing values.

        Returns
        -------
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether an element is an NA value.
            * *True*: An element is a NA value, such as None.
            * *False*: An element is a non-missing value.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["Point (1 1)", "POLYGON ((1 1,2 1,2 2,1 2,1 1))", None, ""])
        >>> s.isna()
        0    False
        1    False
        2     True
        3     True
        dtype: bool
        """
        return super().isna()

    def notna(self):
        """
        Detects existing (non-missing) values.

        Returns
        -------
        Series
            Mask of boolean values for each element in GeoSeries that indicates whether an element is not an NA value.
            * *True*: An element is a non-missing value.
            * *False*: An element is a NA value, such as None.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["Point (1 1)", "POLYGON ((1 1,2 1,2 2,1 2,1 1))", None, ""])
        >>> s.notna()
        0     True
        1     True
        2    False
        3    False
        dtype: bool
        """
        return super().notna()

    # -------------------------------------------------------------------------
    # Geometry related property
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        """
        Tests whether each geometry in the GeoSeries is in valid format, such as WKT and WKB formats.

        Returns
        -------
        Series
        Mask of boolean values for each element in the GeoSeries that indicates whether an element is valid.
            * *True:* The geometry is valid.
            * *False:* The geometry is invalid.

        Examples
        -------
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
        Calculates the length of each geometry in the GeoSeries.

        The ways to calculate the length of geometries are as follows:

        * POINT / MULTIPOINT / POLYGON / MULTIPOLYGON / CURVEPOLYGON / MULTICURVE: 0
        * LINESTRING: Length of a single straight line.
        * MULTILINESTRING: Sum of length of multiple straight lines.
        * CIRCULARSTRING: Length of a single curvilinear line.
        * MULTISURFACE / COMPOUNDCURVE / GEOMETRYCOLLECTION: For a geometry collection among the 3 types, calculates the sum of length of all geometries in the collection.

        Returns
        -------
        Series
            Length of each geometry in the GeoSeries.

        Examples
        -------
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
        Tests whether each geometry in the GeoSeries is simple.

        Here "simple" means that a geometry has no anomalous point, such as a self-intersection or a self-tangency.

        Returns
        -------
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether an element is simple.
            * *True:* The geometry is simple.
            * *False:* The geometry is not simple.

        Examples
        -------
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
        Series
            2D Cartesian (planar) area of each geometry in the GeoSeries.

        Examples
        -------
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
        Returns the type of each geometry in the GeoSeries.

        Returns
        -------
        Series
            The string representations of geometry types. For example, "ST_LINESTRING", "ST_POLYGON", "ST_POINT", and "ST_MULTIPOINT".

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.geom_type
        0      ST_POINT
        1    ST_POLYGON
        dtype: object
        """
        return _property_op(arctern.ST_GeometryType, self)

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
        For each geometry in the GeoSeries, returns the smallest convex geometry that encloses it.

        * For a polygon, the returned geometry is the smallest convex geometry that encloses it.
        * For a geometry collection, the returned geometry is the smallest convex geometry that encloses all geometries in the collection.
        * For a point or line, the returned geometry is the same as the original.

        * For a polygon, the returned geometry is the smallest convex geometry that encloses it.
        * For a geometry collection, the returned geometry is the smallest convex geometry that encloses all geometries in the collection.
        * For a point or line, the returned geometry is the same as the original.

        Returns
        -------
        GeoSeries
            Sequence of convex geometries.

        Examples
        -------
        For geometry collections, such as MULTIPOLYGON, MULTISURFACE, and GEOMETRYCOLLECTION, ``convex_hull`` ignores point and line elements and creates the smallest convex geometry that encloses all polygon elements.

        First, create a MULTIPOLYGON object that contains a concave polygon and a rectangle.

        >>> from arctern import GeoSeries
        >>> from arctern.plot import plot_geometry
        >>> fig, ax = plt.subplots()
        >>> g0 = GeoSeries(["MULTIPOLYGON(((0 0,0 2,1 1,2 2,2 0,0 0)),((2 0,2 2,3 2,3 0,2 0)))"])
        >>> plot_geometry(ax,g0)

        Then, use ``convex_hull`` to get the smallest convex geometry that encloses all geometries in the MULTIPOLYGON object.

        >>> g1 = g0.convex_hull
        >>> fig, ax = plt.subplots()
        >>> plot_geometry(ax,g1)

        Let's see how ``convex_hull`` deals with a GEOMETRYCOLLECTION that contains a semicircle and a rectangle.

        >>> fig, ax = plt.subplots()
        >>> ax.axis('equal')
        >>> g4=GeoSeries(["GEOMETRYCOLLECTION(CURVEPOLYGON(CIRCULARSTRING(1 0,0 1,1 2,1 1,1 0)),polygon((1 0,1 2,2 2,2 0,1 0)))"])
        >>> plot_geometry(ax,g4.curve_to_line())

        Use ``convex_hull`` to get the smallest convex geometry that encloses all geometries in the GEOMETRYCOLLECTION object. Since semicircle and rectangle are convex, the returned convex geometry is just a combination of the two gemetries and looks the same as the original.

        >>> fig, ax = plt.subplots()
        >>> ax.axis('equal')
        >>> print(g4.convex_hull.to_wkt()[0])
        >>> plot_geometry(ax,g4.convex_hull.curve_to_line())
        CURVEPOLYGON (COMPOUNDCURVE (CIRCULARSTRING (1 0,0 1,1 2),(1 2,2 2,2 0,1 0)))

        ``convex_hull`` will not make any changes to POINT, MULTIPOINT, LINESTRING, MULTILINESTRING, and CIRCULARSTRING.

        The GeoSeries ``s1`` below contains a point, a line, and a convex polygon.

        >>> fig, ax = plt.subplots()
        >>> ax.axis('equal')
        >>> s = GeoSeries(["POINT(2 0.5)", "LINESTRING(0 0,3 0.5)",  "POLYGON ((1 1,3 1,3 3,1 3, 1 1))"])
        >>> plot_geometry(ax,s)

        The returned geometries from ``convex_hull`` looks exactly the same as the original.

        >>> fig, ax = plt.subplots()
        >>> ax.axis('equal')
        >>> print(s.convex_hull)
        >>> plot_geometry(ax,s.convex_hull)
        0                    POINT (2.0 0.5)
        1           LINESTRING (0 0,3.0 0.5)
        2    POLYGON ((1 1,1 3,3 3,3 1,1 1))
        dtype: GeoDtype
        """
        return _property_geo(arctern.ST_ConvexHull, self)

    @property
    def npoints(self):
        """
        Returns the number of points for each geometry in the GeoSeries.

        Returns
        -------
        Series
            Number of points of each geometry in the GeoSeries.

        Examples
        -------
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
        Returns the minimum bounding box for each geometry in the GeoSeries.

        The bounding box is a rectangular geometry object, and its edges are parallel to the axes.

        Returns
        -------
        GeoSeries
            Minimum bounding box of each geometry in the GeoSeries.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)", "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"])
        >>> s.envelope
        0                        POINT (1 1)
        1    POLYGON ((1 1,1 3,3 3,3 1,1 1))
        dtype: GeoDtype
        """
        return _property_geo(arctern.ST_Envelope, self)

    @property
    def is_empty(self):
        """
        Returns true if this Geometry is an empty geometry.

        Returns
        --------
        Mask of boolean values for each element in the GeoSeries that indicates whether an element is empty.
            * *True:* The geometry is empty.
            * *False:* The geometry is not empty.

        Examples
        --------
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["LINESTRING EMPTY", "POINT (100 200)"])
        >>> s.is_empty
        0     True
        1    False
        dtype: bool
        """
        return _property_op(arctern.ST_IsEmpty, self).astype(bool, copy=False)

    # -------------------------------------------------------------------------
    # Geometry related unary methods, which return GeoSeries
    # -------------------------------------------------------------------------

    def difference(self, other):
        """
        Returns a geometry that represents that part of geometry slef that does not intersect with geometry other.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to calculate the part of geometry slef that does not intersect with geometry other.
            * If ``other`` is a geometry, this function calculates the difference of each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function calculates the difference of each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        GeoSeries
            A GeoSeries contains geometries that represents that part of geometry slef that does not intersect with geometry other.

        Examples
        --------
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s2 = GeoSeries(["LINESTRING (4 0,6 0)", "POINT (4 0)"])
        >>> s1.difference(s2)
        0    LINESTRING (0 0,4 0)
        1             POINT (6 0)
        dtype: GeoDtype
        """
        return _binary_geo(arctern.ST_Difference, self, other)

    def sym_difference(self, other):
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
            A GeoSeries contains geometries that represents the portions of self and other that do not intersect.

        Examples
        --------
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s2 = GeoSeries(["LINESTRING (4 0,6 0)", "POINT (4 0)"])
        >>> s1.sym_difference(s2)
        0    MULTILINESTRING ((0 0,4 0),(5 0,6 0))
        1                              POINT (6 0)
        dtype: GeoDtype
        """
        return _binary_geo(arctern.ST_SymDifference, self, other)

    def scale(self, factor_x, factor_y):
        """
        Scales the geometry to a new size by multiplying the ordinates with the corresponding factor parameters.

        Parameters
        ----------
        factor_x : double
            The factor to scale coordinate x.
        factor_y : double
            The factor to sclae coordinate y.

        Returns
        -------
        GeoSeries
            A GeoSeries contains geometries with a new size by multiplying the ordinates with the corresponding factor parameters.

        Examples
        --------
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s1.scale(2,2)
        0    LINESTRING (0 0,10 0)
        1    MULTIPOINT (8 0,12 0)
        dtype: GeoDtype
        """
        return _unary_geo(arctern.ST_Scale, self, factor_x, factor_y)

    def affine(self, a, b, d, e, offset_x, offset_y):
        """
        Applies a affine transformation to the geometry to do things like translate, rotate, scale in one step.

        Parameters
        -----------
        a : double
            The parameter in matrix.
        b : double
            The parameter in matrix.
        d : double
            The parameter in matrix.
        e : double
            The parameter in matrix.

        Returns
        --------
        GeoSeries
            A GeoSeries contains geometries which are tranformed by parameters in matrix.

        Examples
        ---------
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["LINESTRING (0 0,5 0)", "MULTIPOINT ((4 0),(6 0))"])
        >>> s1.affine(2,2,2,2,2,2)
        0      LINESTRING (2 2,12 12)
        1    MULTIPOINT (10 10,14 14)
        dtype: GeoDtype
        """
        return _unary_geo(arctern.ST_Affine, self, a, b, d, e, offset_x, offset_y)

    def curve_to_line(self):
        """
        Converts curves in each geometry to approximate linear representation.

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
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))"])
        >>> rst = s.curve_to_line().to_wkt()
        >>> assert str(rst[0]).startswith("POLYGON")
        """
        return _unary_geo(arctern.ST_CurveToLine, self)

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
            raise ValueError(
                "Can not transform geometries without crs. Set crs for this GeoSeries first.")
        if self.crs == crs:
            return self
        return _unary_geo(arctern.ST_Transform, self, self.crs, crs, crs=crs)

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
        -------
        >>> import matplotlib.pyplot as plt
        >>> from arctern import GeoSeries
        >>> from arctern.plot import plot_geometry
        >>> g0 = GeoSeries(["CURVEPOLYGON(CIRCULARSTRING(0 0, 10 0, 10 10, 0 10, 0 0))"])
        >>> g0 = g0.curve_to_line()
        >>> fig, ax = plt.subplots()
        >>> ax.axis('equal')
        >>> ax.grid()
        >>> plot_geometry(ax,g0,facecolor="red",alpha=0.2)
        >>> plot_geometry(ax,g0.simplify(1),facecolor="green",alpha=0.2)
        """
        return _unary_geo(arctern.ST_SimplifyPreserveTopology, self, tolerance)

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
        -------
        >>> import matplotlib.pyplot as plt
        >>> from arctern import GeoSeries
        >>> from arctern.plot import plot_geometry
        >>> g0 = GeoSeries(["CURVEPOLYGON(CIRCULARSTRING(0 0, 10 0, 10 10, 0 10, 0 0))"])
        >>> g0 = g0.curve_to_line()
        >>> fig, ax = plt.subplots()
        >>> ax.axis('equal')
        >>> ax.grid()
        >>> plot_geometry(ax,g0,facecolor=["red"],alpha=0.2)
        >>> plot_geometry(ax,g0.buffer(-2),facecolor=["green"],alpha=0.2)
        >>> plot_geometry(ax,g0.buffer(2),facecolor=["blue"],alpha=0.2)
        """
        return _unary_geo(arctern.ST_Buffer, self, distance)

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
        Creates a valid representation of each geometry in the GeoSeries without losing any of the input vertices.

        If the geometry is already-valid, then nothing will be done. If the geometry can't be made to valid, it will be set to None.

        Returns
        -------
        GeoSeries
            Sequence of valid geometries.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POLYGON ((2 1,3 1,3 2,2 2,2 8,2 1))"])
        >>> s.make_valid()
        0    GEOMETRYCOLLECTION (POLYGON ((2 2,3 2,3 1,2 1,2 2)),LINESTRING (2 2,2 8))
        dtype: GeoDtype
        """
        return _unary_geo(arctern.ST_MakeValid, self)

    def unary_union(self):
        """
        Calculates a geometry that represents the union of all geometries in the GeoSeries.

        Returns
        -------
        GeoSeries
            A GeoSeries that contains only one geometry, which is the union of all geometries in the first GeoSeries.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> p1 = "POINT(1 2)"
        >>> p2 = "POINT(1 1)"
        >>> s = GeoSeries([p1, p2])
        >>> s.unary_union()
        0    MULTIPOINT (1 2,1 1)
        dtype: GeoDtype
        """
        return GeoSeries(arctern.ST_Union_Aggr(self))

    def envelope_aggr(self):
        """
        Returns the minimum bounding box for the union of all geometries in the GeoSeries.

        The bounding box is a rectangular geometry object, and its sides are parallel to the axes.

        Returns
        -------
        GeoSeries
            A GeoSeries that contains only one geometry, which is the minimum bounding box for the union of all geometries in the first GeoSeries.

        Examples
        -------
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
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether they intersect each other.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether it is intersected with the geometries in the first GeoSeries.
            * If ``other`` is a geometry, this function tests the intersection relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the intersection relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it intersects the geometries in ``other``.
            * *True*: The two geometries intersect each other.
            * *False*: The two geometries do not intersect each other.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s1 = GeoSeries(["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((8 0,9 0,9 1,8 1,8 0))"])
        >>> s2 = GeoSeries(["POLYGON((0 0,0 8,8 8,8 0,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"])
        >>> s2.intersects(s1)
        0    True
        1    True
        dtype: bool

        Alternatively, ``other`` can be a geometry in WKB format.

        >>> s2.intersects(s1[0])
        0    True
        1    True
        dtype: bool
        """
        return _binary_op(arctern.ST_Intersects, self, other).astype(bool, copy=False)

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
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, tests whether the first geometry contains the other.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to test whether the geometries in the first GeoSeries contains it.
            * If ``other`` is a geometry, this function tests the "contain" relation between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function tests the "contain" relation between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it contains the geometries in ``other``.
            * *True*: The first geometry contains the other.
            * *False*: The first geometry does not contain the other.

        Examples
        -------
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
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it crosses the geometries in ``other``.
            * *True*: The first geometry crosses the other.
            * *False*: The first geometry does not cross the other.

        Examples
        -------
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
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it equals the geometries in ``other``.
            * *True*: The first geometry equals the other.
            * *False*: The first geometry does not equal the other.

        Examples
        -------
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
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it touches the geometries in ``other``.
            * *True*: The first geometry touches the other.
            * *False*: The first geometry does not touch the other.

        Examples
        -------
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
        Series
            Mask of boolean values for each element in the GeoSeries that indicates whether it overlaps the geometries in ``other``.
            * *True*: The first geometry overlaps the other.
            * *False*: The first geometry does not overlap the other.

        Examples
        -------
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
        For each geometry in the GeoSeries and the corresponding geometry given in ``other``, calculates the minimum 2D Cartesian (planar) distance between them.

        Parameters
        ----------
        other : geometry or GeoSeries
            The geometry or GeoSeries to calculate the distance between it and the geometries in the first GeoSeries.
            * If ``other`` is a geometry, this function calculates the distance between each geometry in the GeoSeries and ``other``.
            * If ``other`` is a GeoSeries, this function calculates the distance between each geometry in the GeoSeries and the geometry with the same index in ``other``.

        Returns
        -------
        Series
            Distance between each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Examples
        -------
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
        Series
            Spherical distance between each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Notes
        -------
        Only the longitude and latitude coordinate reference system ("EPSG:4326") can be used to calculate spherical distance.

        Examples
        -------
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
        Series
            Hausdorff distance between each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Examples
        -------
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
            A GeoSeries that contains only one geometry, which is the intersection of each geometry in the GeoSeries and the corresponding geometry given in ``other``.

        Examples
        -------
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
        Transforms all geometries in the GeoSeries to `WKT <https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry>`_ strings.

        Returns
        -------
        Series
            Sequence of geometries in WKT format.

        Examples
        -------
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
        Transforms all geometries in the GeoSeries to `WKB <https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry#Well-known_binary>`_ strings.

        Returns
        -------
        Series
            Sequence of geometries in WKB format.

        Examples
        -------
        >>> from arctern import GeoSeries
        >>> s = GeoSeries(["POINT(1 1)"])
        >>> s.to_wkb()
        0    b'\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?'
        dtype: object
        """
        return _property_op(lambda x: x, self)

    def as_geojson(self):
        """
        Transforms all geometries in the GeoSeries to GeoJSON strings.

        Returns
        -------
        Series
            Sequence of geometries in GeoJSON format.

        Examples
        -------
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
        Transforms an arctern.GeoSeries object to a geopandas.GeoSeries object.

        Returns
        -------
        geopandas.GeoSeries
            A geopandas.GeoSeries object.

        Examples
        -------
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
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, ``"EPSG:4326"``.

        Returns
        -------
        GeoSeries
            Sequence of rectangular POLYGON objects within the given spatial range.

        Examples
        -------
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
        Constructs POINT objects based on the given coordinates.

        ``x``and ``y`` are Series so that points can be created in batch. The number of values in the two Series should be the same.

        Suppose that the demension of ``x`` is *N*, the returned GeoSeries of this function should contains *N* points. The position of the *i*th point is defined by its coordinates *(x, y)*.

        Parameters
        ----------
        x : Series
            X coordinates of points.
        y : Series
            Y coordinates of points.
        crs : str, optional
            A string representation of Coordinate Reference System (CRS).
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, ``"EPSG:4326"``.

        Returns
        -------
        GeoSeries
            Sequence of POINT objects.

        Examples
        -------
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
        Constructs geometries from GeoJSON strings.

        ``json`` is Series so that geometries can be created in batch.

        Parameters
        ----------
        json : Series
            String representations of geometries in JSON format.
        crs : str, optional
            A string representation of Coordinate Reference System (CRS).
            The string is made up of an authority code and a SRID (Spatial Reference Identifier), for example, ``"EPSG:4326"``.

        Returns
        -------
        GeoSeries
            Sequence of geometries.

        Examples
        -------
        >>> from pandas import Series
        >>> from arctern import GeoSeries
        >>> json = Series(['{"type":"LineString","coordinates":[[1,2],[4,5],[7,8]]}'])
        >>> GeoSeries.geom_from_geojson(json)
        0    LINESTRING (1 2,4 5,7 8)
        dtype: GeoDtype
        """
        crs = _validate_crs(crs)
        return cls(arctern.ST_GeomFromGeoJSON(json), crs=crs)

    @classmethod
    def from_geopandas(cls, data):
        """
        Constructs an arctern.GeoSeries object from a geopandas.GeoSeries object.

        Parameters
        ----------
        data : geopandas.GeoSeries
            A geopandas.GeoSeries object.

        Returns
        -------
        arctern.GeoSeries
            An arctern.GeoSeries object.

        Examples
        -------
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point, Polygon
        >>> from arctern import GeoSeries
        >>> gpd_s = gpd.GeoSeries([Point(1,1), Polygon(((1,1), (1,2), (2,3), (1,1)))])
        >>> arctern_s = GeoSeries.from_geopandas(gpd_s)
        >>> arctern_s
        0                    POINT (1 1)
        1    POLYGON ((1 1,1 2,2 3,1 1))
        dtype: GeoDtype
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
