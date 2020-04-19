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
# pylint: disable=too-many-lines

__all__ = [
    "ST_Point",
    "ST_Intersection",
    "ST_IsValid",
    "ST_PrecisionReduce",
    "ST_Equals",
    "ST_Touches",
    "ST_Overlaps",
    "ST_Crosses",
    "ST_IsSimple",
    "ST_GeometryType",
    "ST_MakeValid",
    "ST_SimplifyPreserveTopology",
    "ST_PolygonFromEnvelope",
    "ST_Contains",
    "ST_Intersects",
    "ST_Within",
    "ST_Distance",
    "ST_DistanceSphere",
    "ST_Area",
    "ST_Centroid",
    "ST_Length",
    "ST_HausdorffDistance",
    "ST_ConvexHull",
    "ST_NPoints",
    "ST_Envelope",
    "ST_Buffer",
    "ST_Union_Aggr",
    "ST_Envelope_Aggr",
    "ST_Transform",
    "ST_CurveToLine",
    "ST_GeomFromGeoJSON",
    "ST_GeomFromText",
    "ST_AsText",
    "ST_AsGeoJSON",
    "point_map",
    "weighted_point_map",
    "heat_map",
    "choropleth_map",
    "icon_viz",
    "projection",
    "transform_and_projection",
    "wkt2wkb",
    "wkb2wkt",
]

import base64
from . import arctern_core_

def arctern_udf(*arg_types):
    def decorate(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*warpper_args):
            import pandas as pd
            pd_series_type = type(pd.Series([None]))
            array_len = 1
            for arg in warpper_args:
                if isinstance(arg, pd_series_type):
                    array_len = len(arg)
                    break
            func_args = []
            func_arg_idx = 0
            for arg_type in arg_types:
                if arg_type is None:
                    func_args.append(warpper_args[func_arg_idx])
                else:
                    assert isinstance(arg_type, str)
                    if len(arg_type) == 0:
                        func_args.append(warpper_args[func_arg_idx])
                    elif isinstance(warpper_args[func_arg_idx], pd_series_type):
                        assert len(warpper_args[func_arg_idx]) == array_len
                        func_args.append(warpper_args[func_arg_idx])
                    else:
                        if arg_type == 'binary':
                            arg_type = 'object'
                        arg = pd.Series([warpper_args[func_arg_idx] for _ in range(array_len)], dtype=arg_type)
                        func_args.append(arg)
                func_arg_idx = func_arg_idx + 1
            while func_arg_idx < len(warpper_args):
                func_args.append(warpper_args[func_arg_idx])
                func_arg_idx = func_arg_idx + 1
            return func(*func_args)
        return wrapper
    return decorate

@arctern_udf('double', 'double')
def ST_Point(x, y):
    """
    Construct Point geometries according to the coordinates.

    :type x: pandas.Series.float64
    :param x: Abscissa of the point.

    :type y: pandas.Series.float64
    :param y: Ordinate of the point.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series([1.3, 2.5])
      >>> data2 = pandas.Series([1.3, 2.5])
      >>> string_ptr = arctern.ST_AsText(arctern.ST_Point(data1, data2))
      >>> print(string_ptr)
          0    POINT (1.3 3.8)
          1    POINT (2.5 4.9)
          dtype: object
    """
    import pyarrow as pa
    arr_x = pa.array(x, type='double')
    arr_y = pa.array(y, type='double')
    rs = arctern_core_.ST_Point(arr_x, arr_y)
    return rs.to_pandas()


@arctern_udf('string')
def ST_GeomFromGeoJSON(json):
    """
    Constructs a geometry object from the GeoJSON representation.

    :type json: pandas.Series.object
    :param json: Geometries organized as json

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["{\"type\":\"LineString\",\"coordinates\":[[1,2],[4,5],[7,8]]}"])
      >>> string_ptr = arctern.ST_AsText(arctern.ST_GeomFromGeoJSON(data))
      >>> print(string_ptr)
          0    LineString (1 2,4 5,7 8)
          dtype: object
    """
    import pyarrow as pa
    geo = pa.array(json, type='string')
    rs = arctern_core_.ST_GeomFromGeoJSON(geo)
    return rs.to_pandas()


@arctern_udf('string')
def ST_GeomFromText(text):
    """
    Constructs a geometry object from the OGC Well-Known text representation.

    :type json: pandas.Series.object
    :param json: Geometries organized as wkt

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((0 0,0 1,1 1,1 0,0 0))"])
      >>> string_ptr = arctern.ST_AsText(arctern.ST_GeomFromText(data))
      >>> print(string_ptr)
          0    POLYGON ((0 0,0 1,1 1,1 0,0 0))
          dtype: object
    """
    import pyarrow as pa
    geo = pa.array(text, type='string')
    rs = arctern_core_.ST_GeomFromText(geo)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_AsText(text):
    """
    Returns the Well-Known Text representation of the geometry.

    :type text: pandas.Series.object
    :param text: Geometries organized as WKB.

    :return: Geometries organized as WKT.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((0 0,0 1,1 1,1 0,0 0))"])
      >>> string_ptr = arctern.ST_AsText(arctern.ST_GeomFromText(data))
      >>> print(string_ptr)
          0    POLYGON ((0 0,0 1,1 1,1 0,0 0))
          dtype: object
    """
    import pyarrow as pa
    geo = pa.array(text, type='binary')
    rs = arctern_core_.ST_AsText(geo)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_AsGeoJSON(text):
    """
    Returns the GeoJSON representation of the geometry.

    :type text: pyarrow.array.string
    :param text: Geometries organized as WKB.

    :return: Geometries organized as GeoJSON.
    :rtype: pyarrow.array.string

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((0 0,0 1,1 1,1 0,0 0))"])
      >>> string_ptr = arctern.ST_AsGeoJSON(arctern.ST_GeomFromText(data))
      >>> print(string_ptr)
          0    { "type": "Polygon", "coordinates": [ [ [ 0.0, 0.0 ], [ 0.0, 1.0 ], [ 1.0, 1.0 ], [ 1.0, 0.0 ], [ 0.0, 0.0 ] ] ] }
          dtype: object
    """
    import pyarrow as pa
    geo = pa.array(text, type='binary')
    rs = arctern_core_.ST_AsGeoJSON(geo)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Intersection(left, right):
    """
    Calculate the point set intersection of geometries.

    For every (left, right) pair with the same offset value in left and right,
    calculate a geometry that represents their point set intersection.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> data2 = pandas.Series(["POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
      >>> string_ptr = arctern.ST_AsText(arctern.ST_Point(data1, data2))
      >>> print(string_ptr)
          0    LINESTRING (2 2,2 1)
          dtype: object
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Intersection(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_IsValid(geos):
    """
    For each item in geometries, check if it is of valid geometry format.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POINT (1.3 2.6)", "POINT (2.6 4.7)"])
      >>> rst = arctern.ST_IsValid(arctern.ST_GeomFromText(data))
      >>> print(rst)
          0    true
          1    true
          dtype: bool
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_IsValid(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary', '')
def ST_PrecisionReduce(geos, precision):
    """
    Reduce the precision of geometry.

    For every geometry in geometries, reduce the decimal places of its coordinates
    to the given number. The last decimal place will be rounded.

    Note, the operation is performed NOT in "inplace" manner, i.e., new geometries
    in arrow::Array format will be construted and extra memory will be allocated.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKT.

    :type precision: uint32
    :param geos: The number to reduce the decimals places to.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POINT (1.333 2.666)", "POINT (2.655 4.447)"])
      >>> rst = arctern.arctern.ST_AsText(arctern.ST_PrecisionReduce(arctern.ST_GeomFromText(data), 3))
      >>> print(rst)
          0    POINT (1.33 2.67)
          1    POINT (2.66 4.45)
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_PrecisionReduce(arr_geos, precision)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Equals(left, right):
    """
    Check whether geometries are "spatially equal".

    For every (left, right) pair with the same offset value in left and right, check
    if they are "spatially equal". "Spatially equal" here means two geometries represent
    the same geometry structure.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
      >>> rst = arctern.ST_Equals(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
      >>> print(rst)
          0    true
          1    false
          dtype: bool
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Equals(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Touches(left, right):
    """
    Check whether geometries "touch".

    For every (left, right) pair with the same offset value in left and right, check
    if they "touch". "Touch" here means two geometries have common points, and the
    common points locate only on their boundaries.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
      >>> rst = arctern.ST_Touches(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
      >>> print(rst)
          0    false
          1    true
          dtype: bool
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Touches(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Overlaps(left, right):
    """
    Check whether geometries "spatially overlap".

    For every (left, right) pair with the same offset value in left and right, check
    if they "spatially overlap". "Spatially overlap" here means two geometries
    intersect but one does not completely contain another.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
      >>> rst = arctern.ST_Touches(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
      >>> print(rst)
          0    false
          1    false
          dtype: bool
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Overlaps(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Crosses(left, right):
    """
    Check whether geometries "spatially cross".

    For every (left, right) pair with the same offset value in left and right, check
    if they "spatially cross". "Spatially cross" here means two the geometries have
    some, but not all interior points in common. The intersection of the interiors of
    the geometries must not be the empty set and must have a dimensionality less than
    the maximum dimension of the two input geometries.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
      >>> rst = arctern.ST_Touches(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
      >>> print(rst)
          0    false
          1    false
          dtype: bool
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Crosses(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_IsSimple(geos):
    """
    Check whether geometry is "simple".

    For every geometry in geometries, check if it is "simple". "Simple" here means
    that a geometry has no anomalous geometric points such as self intersection or
    self tangency.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> rst = arctern.ST_IsSimple(arctern.ST_GeomFromText(data))
      >>> print(rst)
          0    true
          1    true
          dtype: bool
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_IsSimple(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_GeometryType(geos):
    """
    For each geometry in geometries, return a string that indicates is type.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> rst = arctern.ST_AsText(arctern.ST_GeometryType(arctern.ST_GeomFromText(data)))
      >>> print(rst)
          0    ST_POLYGON
          1    ST_POLYGON
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_GeometryType(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_MakeValid(geos):
    """
    For every geometry in geometries, create a valid representation of it without
    losing any of the input vertices. Already-valid geometries won't have further
    intervention. This function returns geometries which are validated. Note, new
    geometries are construted in arrow::Array format, so extra memory will be allocated.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((2 1,3 1,3 2,2 2,2 8,2 1))"])
      >>> rst = arctern.ST_AsText(arctern.ST_MakeValid(arctern.ST_GeomFromText(data)))
      >>> print(rst)
          0    GEOMETRYCOLLECTION (POLYGON ((2 2,3 2,3 1,2 1,2 2)),LINESTRING (2 2,2 8))
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_MakeValid(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    """
    For each geometry in geometries create a "simplified" version for it according
    to the precision that parameter tolerance specifies.

    Note simplified geometries with be construted in arrow::Array format, so extra
    memory will be allocated.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :type distance_tolerance: double
    :param distance_tolerance: The precision of the simplified geometry.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> rst = arctern.ST_AsText(arctern.ST_SimplifyPreserveTopology(arctern.ST_GeomFromText(data), 10000))
      >>> print(rst)
          0    POLYGON ((1 1,1 2,2 2,2 1,1 1))
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_SimplifyPreserveTopology(arr_geos, distance_tolerance)
    return rs.to_pandas()

@arctern_udf('double', 'double', 'double', 'double')
def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    """
    Construct polygon(rectangle) geometries from arr_min_x, arr_min_y, arr_max_x,
    arr_max_y. The edges of polygon are parallel to coordinate axis.

    :type min_x: pandas.Series.float64
    :param min_x: The x axis coordinates of the lower left vertical of the rectangles.

    :type min_y: pandas.Series.float64
    :param min_y: The y axis coordinates of the lower left vertical of the rectangles.

    :type max_x: pandas.Series.float64
    :param max_x: The x axis coordinates of the upper right vertical of the rectangles.

    :type max_y: pandas.Series.float64
    :param max_y: The y axis coordinates of the upper right vertical of the rectangles.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> x_min = pandas.Series([0.0])
      >>> x_max = pandas.Series([1.0])
      >>> y_min = pandas.Series([0.0])
      >>> y_max = pandas.Series([1.0])
      >>> rst = arctern.ST_AsText(rctern.ST_PolygonFromEnvelope(x_min, y_min, x_max, y_max))
      >>> print(rst)
          0    POLYGON ((0 0,0 1,1 1,1 0,0 0))
          dtype: object
    """
    import pyarrow as pa
    arr_min_x = pa.array(min_x, type='double')
    arr_min_y = pa.array(min_y, type='double')
    arr_max_x = pa.array(max_x, type='double')
    arr_max_y = pa.array(max_y, type='double')
    rs = arctern_core_.ST_PolygonFromEnvelope(arr_min_x, arr_min_y, arr_max_x, arr_max_y)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Contains(left, right):
    """
    Check whether a geometry contain another geometry.

    For every (left, right) pair with the same offset value in left and right, check
    if left_geometry "contains" right_geometry. Left "contains" right means no points
    of right_geometry lie in the exterior of left_geometry and at least one point of
    the interior of right_geometry lies in the interior of left_geometry.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON((0 0,1 0,1 1,0 1,0 0))","POLYGON((8 0,9 0,9 1,8 1,8 0))"])
      >>> data2 = pandas.Series(["POLYGON((0 0,0 8,8 8,8 0,0 0))","POLYGON((0 0,0 8,8 8,8 0,0 0))"])
      >>> rst = arctern.ST_Contains(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
      >>> print(rst)
          0    true
          1    false
          dtype: bool
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Contains(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Intersects(left, right):
    """
    Check whether two geometries intersect.

    For every (left, right) pair with the same offset value in left and right, check
    if left and right shares any portion of space.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON((0 0,1 0,1 1,0 1,0 0))","POLYGON((8 0,9 0,9 1,8 1,8 0))"])
      >>> data2 = pandas.Series(["POLYGON((0 0,0 8,8 8,8 0,0 0))","POLYGON((0 0,0 8,8 8,8 0,0 0))"])
      >>> rst = arctern.ST_Intersects(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
      >>> print(rst)
          0    true
          1    true
          dtype: bool
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Intersects(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Within(left, right):
    """
    Check whether a geometry is within another geometry.

    For every (left, right) pair with the same offset value in left and right, check
    if left is "within" right. Left "within" right means no points of left lie in the
    exterior of right and at least one point of the interior of left lies in the interior
    of right.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of booleans.
    :rtype: pandas.Series.bool

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON((0 0,1 0,1 1,0 1,0 0))","POLYGON((8 0,9 0,9 1,8 1,8 0))"])
      >>> data1 = pandas.Series(["POLYGON((0 0,0 8,8 8,8 0,0 0))","POLYGON((0 0,0 8,8 8,8 0,0 0))"])
      >>> rst = arctern.ST_Within(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
      >>> print(rst)
          0    false
          1    false
          dtype: bool
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Within(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_Distance(left, right):
    """
    Calculate the distance between two geometries.

    For every (left, right) pair with the same offset value in left and right,
    calculates the minimum 2D Cartesian (planar) distance between left and right.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> import pandas
      >>> import arctern
      >>> p11 = "LINESTRING(9 0,9 2)"
      >>> p12 = "POINT(10 2)"
      >>> data1 = pandas.Series([p11, p12])
      >>> p21 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
      >>> p22 = "POLYGON((0 0,0 8,8 8,8 0,0 0))"
      >>> data2 = pandas.Series([p21, p22])
      >>> rst = arctern.ST_Distance(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
      >>> print(rst)
          0    1.0
          1    2.0
          dtype: float64
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Distance(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_DistanceSphere(left, right):
    """
    Returns minimum distance in meters between two lon/lat points.
    Uses a spherical earth and radius derived from the spheroid defined by the SRID.

    For every (left, right) pair with the same offset value in left and right,
    calculates the minimum spherical distance between left and right.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> import pandas
      >>> import arctern
      >>> p11 = "POINT(10 2)"
      >>> p12 = "POINT(10 2)"
      >>> data1 = pandas.Series([p11, p12])
      >>> p21 = "POINT(10 2)"
      >>> p22 = "POINT(10 2)"
      >>> data2 = pandas.Series([p21, p22])
      >>> rst = arctern.ST_DistanceSphere(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
      >>> print(rst)
          0    1.0
          1    2.0
          dtype: float64
    """
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_DistanceSphere(arr_left, arr_right)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_Area(geos):
    """
    Calculate the area of geometry.

    For every geometry in geometries, calculate the 2D Cartesian (planar) area
    of geometry.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = ["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_Area(arctern.ST_GeomFromText(data1))
      >>> print(rst)
          0     1.0
          1    64.0
          dtype: float64
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Area(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_Centroid(geos):
    """
    Compute the centroid of geometry.

    For every geometry in geometries, compute the controid point of geometry.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = ["POLYGON((0 0,1 0,1 1,0 1,0 0))", "POLYGON((0 0,0 8,8 8,8 0,0 0))"]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_AsText(arctern.ST_Centroid(arctern.ST_GeomFromText(data)))
      >>> print(rst)
          0    POINT (0.5 0.5)
          1    POINT (4 4)
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Centroid(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_Length(geos):
    """
    Calculate the length of linear geometries.

    For every geometry in geometries, calculate the length of geometry.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = ["LINESTRING(0 0,0 1)", "LINESTRING(1 1,1 4)"]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_Length(arctern.ST_GeomFromText(data))
      >>> print(rst)
          0    1.0
          1    3.0
          dtype: float64
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Length(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary', 'binary')
def ST_HausdorffDistance(geo1, geo2):
    """
    Returns the Hausdorff distance between two geometries, a measure of how similar
    or dissimilar 2 geometries are.

    Implements algorithm for computing a distance metric which can be thought of as
    the "Discrete Hausdorff Distance". This is the Hausdorff distance restricted to
    discrete points for one of the geometries. Wikipedia article on Hausdorff distance
    Martin Davis note on how Hausdorff Distance calculation was used to prove
    correctness of the CascadePolygonUnion approach.

    When densifyFrac is specified, this function performs a segment densification before
    computing the discrete hausdorff distance. The densifyFrac parameter sets the fraction
    by which to densify each segment. Each segment will be split into a number of equal-length
    subsegments, whose fraction of the total length is closest to the given fraction.

    Units are in the units of the spatial reference system of the geometries.

    :type left: pandas.Series.object
    :param left: Geometries organized as WKB.

    :type right: pandas.Series.object
    :param right: Geometries organized as WKB.

    :return: An array of double.
    :rtype: pandas.Series.float64

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = ["POLYGON((0 0 ,0 1, 1 1, 1 0, 0 0))", "POINT(0 0)"]
      >>> data2 = ["POLYGON((0 0 ,0 2, 1 1, 1 0, 0 0))", "POINT(0 1)"]
      >>> data1 = pandas.Series(data1)
      >>> data2 = pandas.Series(data2)
      >>> rst = arctern.ST_HausdorffDistance(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
      >>> print(rst)
          0    1.0
          1    1.0
          dtype: float64
    """
    import pyarrow as pa
    arr1 = pa.array(geo1, type='binary')
    arr2 = pa.array(geo2, type='binary')
    rs = arctern_core_.ST_HausdorffDistance(arr1, arr2)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_ConvexHull(geos):
    """
    Compute the convex hull of geometry.

    Compute the smallest convex geometry that encloses all geometries for a geometry
    in geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = ["POINT (1.1 101.1)"]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_AsText(arctern.ST_ConvexHull(arctern.ST_GeomFromText(data)))
      >>> print(rst)
          0    POINT (1.1 101.1)
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_ConvexHull(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_NPoints(geos):
    """
    Calculates the points number for every geometry in geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return : An array of int64.
    :rtype : pandas.Series.int64

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = ["LINESTRING(1 1,1 4)"]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_NPoints(arctern.ST_GeomFromText(data))
      >>> print(rst)
          0    2
          dtype: int64
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_NPoints(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_Envelope(geos):
    """
    Compute the double-precision minimum bounding box geometry for every geometry in geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> p1 = "point (10 10)"
      >>> p2 = "linestring (0 0 , 0 10)"
      >>> p3 = "linestring (0 0 , 10 0)"
      >>> p4 = "linestring (0 0 , 10 10)"
      >>> p5 = "polygon ((0 0, 10 0, 10 10, 0 10, 0 0))"
      >>> p6 = "multipoint (0 0, 10 0, 5 5)"
      >>> p7 = "multilinestring ((0 0, 5 5), (6 6, 6 7, 10 10))"
      >>> p8 = "multipolygon (((0 0, 10 0, 10 10, 0 10, 0 0), (11 11, 20 11, 20 20, 20 11, 11 11)))"
      >>> data = [p1, p2, p3, p4, p5, p6, p7, p8]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_AsText(arctern.ST_Envelope(arctern.ST_GeomFromText(data)))
      >>> print(rst)
          0    POINT (10 10)
          1    LINESTRING (0 0,0 10)
          2    LINESTRING (0 0,10 0)
          3    POLYGON ((0 0,0 10,10 10,10 0,0 0))
          4    POLYGON ((0 0,0 10,10 10,10 0,0 0))
          5    POLYGON ((0 0,0 5,10 5,10 0,0 0))
          6    POLYGON ((0 0,0 10,10 10,10 0,0 0))
          7    POLYGON ((0 0,0 20,20 20,20 0,0 0))
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Envelope(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_Buffer(geos, distance):
    """
    Returns a geometry that represents all points whose distance from this geos is
    less than or equal to distance.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :type distance: int64
    :param distance: The maximum distance of the returned geometry from the given geometry.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = ["POINT (0 1)"]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_AsText(arctern.ST_Buffer(arctern.ST_GeomFromText(data), 0))
      >>> print(rst)
          0    POLYGON EMPTY
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Buffer(arr_geos, distance)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_Union_Aggr(geos):
    """
    This function returns a MULTI geometry or NON-MULTI geometry from a set of geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometry organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
      >>> p2 = "POLYGON ((5 1,7 1,7 2,5 2,5 1))"
      >>> data = pandas.Series([p1, p2])
      >>> rst = arctern.ST_AsText(arctern.ST_Union_Aggr(arctern.ST_GeomFromText(data)))
      >>> print(rst)
          0    MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)),((5 1,7 1,7 2,5 2,5 1)))
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Union_Aggr(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_Envelope_Aggr(geos):
    """
    Compute the double-precision minimum bounding box geometry for every geometry in geometries,
    then returns a MULTI geometry or NON-MULTI geometry from a set of geometries.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometry organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))"
      >>> p2 = "POLYGON ((5 1,7 1,7 2,5 2,5 1))"
      >>> data = pandas.Series([p1, p2])
      >>> rst = arctern.ST_AsText(arctern.ST_Envelope_Aggr(arctern.ST_GeomFromText(data)))
      >>> print(rst)
          0    POLYGON ((0 0,0 4,7 4,7 0,0 0))
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Envelope_Aggr(arr_geos)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_Transform(geos, src, dst):
    """
    Returns a new geometry with its coordinates transformed to a different spatial
    reference system.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :type src: string
    :param src: The current srid of geometries.

    :type dst: string
    :param dst: The target srid of geometries tranfrom to.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = ["POINT (10 10)"]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_AsText(arctern.ST_Transform(arctern.ST_GeomFromText(data), "EPSG:4326", "EPSG:3857"))
      >>> wkt = rst[0]
      >>> rst_point = ogr.CreateGeometryFromWkt(str(wkt))
      >>> assert abs(rst_point.GetX() - 1113194.90793274 < 0.01)
      >>> assert abs(rst_point.GetY() - 1118889.97485796 < 0.01)
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    src = bytes(src, encoding="utf8")
    dst = bytes(dst, encoding="utf8")

    rs = arctern_core_.ST_Transform(arr_geos, src, dst)
    return rs.to_pandas()

@arctern_udf('binary')
def ST_CurveToLine(geos):
    """
    Converts a CIRCULAR STRING to regular LINESTRING or CURVEPOLYGON to POLYGON or
    MULTISURFACE to MULTIPOLYGON. Useful for outputting to devices that can't support
    CIRCULARSTRING geometry types.

    :type geos: pandas.Series.object
    :param geos: Geometries organized as WKB.

    :return: Geometries organized as WKB.
    :rtype: pandas.Series.object

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = ["CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0))"]
      >>> data = pandas.Series(data)
      >>> rst = arctern.ST_CurveToLine(arctern.ST_GeomFromText(data))
      >>> assert str(rst[0]).startswith("POLYGON")
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_CurveToLine(arr_geos)
    return rs.to_pandas()

def point_map(vega, points):
    import pyarrow as pa
    array_points = pa.array(points, type='binary')
    vega_string = vega.build().encode('utf-8')
    rs = arctern_core_.point_map(vega_string, array_points)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def weighted_point_map(vega, points, **kwargs):
    import pyarrow as pa
    color_weights = kwargs.get('color_weights', None)
    size_weights = kwargs.get('size_weights', None)
    vega_string = vega.build().encode('utf-8')

    array_points = pa.array(points, type='binary')
    if (color_weights is None and size_weights is None):
        rs = arctern_core_.weighted_point_map(vega_string, array_points)
    elif (color_weights is not None and size_weights is not None):
        if isinstance(color_weights[0], float):
            arr_c = pa.array(color_weights, type='double')
        else:
            arr_c = pa.array(color_weights, type='int64')

        if isinstance(size_weights[0], float):
            arr_s = pa.array(size_weights, type='double')
        else:
            arr_s = pa.array(size_weights, type='int64')
        rs = arctern_core_.weighted_color_size_point_map(vega_string, array_points, arr_c, arr_s)
    elif (color_weights is None and size_weights is not None):
        if isinstance(size_weights[0], float):
            arr_s = pa.array(size_weights, type='double')
        else:
            arr_s = pa.array(size_weights, type='int64')
        rs = arctern_core_.weighted_size_point_map(vega_string, array_points, arr_s)
    else:
        if isinstance(color_weights[0], float):
            arr_c = pa.array(color_weights, type='double')
        else:
            arr_c = pa.array(color_weights, type='int64')
        rs = arctern_core_.weighted_color_point_map(vega_string, array_points, arr_c)

    return base64.b64encode(rs.buffers()[1].to_pybytes())

def heat_map(vega, points, weights):
    import pyarrow as pa
    array_points = pa.array(points, type='binary')
    vega_string = vega.build().encode('utf-8')

    if isinstance(weights[0], float):
        arr_c = pa.array(weights, type='double')
    else:
        arr_c = pa.array(weights, type='int64')

    rs = arctern_core_.heat_map(vega_string, array_points, arr_c)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def choropleth_map(vega, region_boundaries, weights):
    import pyarrow as pa
    arr_wkb = pa.array(region_boundaries, type='binary')
    vega_string = vega.build().encode('utf-8')

    if isinstance(weights[0], float):
        arr_c = pa.array(weights, type='double')
    else:
        arr_c = pa.array(weights, type='int64')
    rs = arctern_core_.choropleth_map(vega_string, arr_wkb, arr_c)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def icon_viz(vega, points):
    import pyarrow as pa
    array_points = pa.array(points, type='binary')
    vega_string = vega.build().encode('utf-8')
    rs = arctern_core_.icon_viz(vega_string, array_points)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def projection(geos, bottom_right, top_left, height, width):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    bounding_box_min = bytes(bottom_right, encoding="utf8")
    bounding_box_max = bytes(top_left, encoding="utf8")
    rs = arctern_core_.projection(arr_geos, bounding_box_min, bounding_box_max, height, width)
    return rs.to_pandas()

def transform_and_projection(geos, src_rs, dst_rs, bottom_right, top_left, height, width):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    src = bytes(src_rs, encoding="utf8")
    dst = bytes(dst_rs, encoding="utf8")
    bounding_box_min = bytes(bottom_right, encoding="utf8")
    bounding_box_max = bytes(top_left, encoding="utf8")
    rs = arctern_core_.transform_and_projection(arr_geos, src, dst, bounding_box_min, bounding_box_max, height, width)
    return rs.to_pandas()

def wkt2wkb(arr_wkt):
    import pyarrow as pa
    wkts = pa.array(arr_wkt, type='string')
    rs = arctern_core_.wkt2wkb(wkts)
    return rs.to_pandas()

def wkb2wkt(arr_wkb):
    import pyarrow as pa
    wkbs = pa.array(arr_wkb, type='binary')
    rs = arctern_core_.wkb2wkt(wkbs)
    return rs.to_pandas()
