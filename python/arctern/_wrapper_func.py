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
    "point_map_layer",
    "weighted_point_map_layer",
    "heat_map_layer",
    "choropleth_map_layer",
    "icon_viz_layer",
    "fishnet_map_layer",
    "projection",
    "transform_and_projection",
    "wkt2wkb",
    "wkb2wkt",
    "version"
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

def arctern_caller(func, *func_args):
    import pyarrow
    num_chunks = 1
    for arg in func_args:
        # pylint: disable=c-extension-no-member
        if isinstance(arg, pyarrow.lib.ChunkedArray):
            num_chunks = len(arg.chunks)
            break

    if num_chunks <= 1:
        result = func(*func_args)
        return result.to_pandas()

    result_total = None
    for chunk_idx in range(num_chunks):
        args = []
        for arg in func_args:
            # pylint: disable=c-extension-no-member
            if isinstance(arg, pyarrow.lib.ChunkedArray):
                args.append(arg.chunks[chunk_idx])
            else:
                args.append(arg)
        result = func(*args)
        if result_total is None:
            result_total = result.to_pandas()
        else:
            result_total = result_total.append(result.to_pandas(), ignore_index=True)
    return result_total

def _to_arrow_array_list(arrow_array):
    if hasattr(arrow_array, 'chunks'):
        return list(arrow_array.chunks)
    return [arrow_array]

def _to_pandas_series(array_list):
    result = None

    for array in array_list:
        if isinstance(array, list):
            for arr in array:
                if result is None:
                    result = arr.to_pandas()
                else:
                    result = result.append(arr.to_pandas(), ignore_index=True)
        else:
            if result is None:
                result = array.to_pandas()
            else:
                result = result.append(array.to_pandas(), ignore_index=True)
    return result

@arctern_udf('double', 'double')
def ST_Point(x, y):
    """
    Construct Point geometries according to the coordinates.

    :type x: Series(dtype: float64)
    :param x: Abscissa of the point.

    :type y: Series(dtype: float64)
    :param y: Ordinate of the point.

    :rtype: Series(dtype: object)
    :return: Point in WKB form.

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
    arr_x = _to_arrow_array_list(arr_x)
    arr_y = _to_arrow_array_list(arr_y)
    result = arctern_core_.ST_Point(arr_x, arr_y)
    return _to_pandas_series(result)

@arctern_udf('string')
def ST_GeomFromGeoJSON(json):
    """
    Construct geometry from the GeoJSON representation.

    :type json: Series(dtype: object)
    :param json: Geometries in json format.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

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
    geo = _to_arrow_array_list(geo)
    result = [arctern_core_.ST_GeomFromGeoJSON(g) for g in geo]
    return _to_pandas_series(result)

@arctern_udf('string')
def ST_GeomFromText(text):
    """
    Transform the representation of geometry from WKT to WKB.

    :type json: Series(dtype: object)
    :param json: Geometries in WKT form.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

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
    geo = _to_arrow_array_list(geo)
    result = [arctern_core_.ST_GeomFromText(g) for g in geo]
    return _to_pandas_series(result)

@arctern_udf('binary')
def ST_AsText(text):
    """
    Transform the representation of geometry from WKB to WKT.

    :type text: Series(dtype: object)
    :param text: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: Geometries in WKT form.

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
    geo = _to_arrow_array_list(geo)
    result = [arctern_core_.ST_AsText(g) for g in geo]
    return _to_pandas_series(result)

@arctern_udf('binary')
def ST_AsGeoJSON(text):
    """
    Return the GeoJSON representation of the geometry.

    :type text: Series(dtype: object)
    :param text: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: Geometries in GeoJSON format.

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
    geo = _to_arrow_array_list(geo)
    result = [arctern_core_.ST_AsGeoJSON(g) for g in geo]
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_Intersection(geo1, geo2):
    """
    Calculate the point set intersection of two geometry objects.

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

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
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Intersection(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary')
def ST_IsValid(geos):
    """
    Check if geometry is of valid geometry format.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry is valid.

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
    return arctern_caller(arctern_core_.ST_IsValid, arr_geos)

@arctern_udf('binary', '')
def ST_PrecisionReduce(geos, precision):
    """
    For the coordinates of the geometry, reduce the number of significant digits
    to the given number. The last decimal place will be rounded.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :type precision: int
    :param precision: The number to of ignificant digits.

    :rtype: Series(dtype: object)
    :return: Geometry with reduced precision.

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
    return arctern_caller(arctern_core_.ST_PrecisionReduce, arr_geos, precision)

@arctern_udf('binary', 'binary')
def ST_Equals(geo1, geo2):
    """
    Check whether geometries are "spatially equal". "Spatially equal" here means two geometries represent
    the same geometry structure.

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry "geo1" equals geometry "geo2".

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
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Equals(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_Touches(geo1, geo2):
    """
    Check whether geometries "touch". "Touch" here means two geometries have common points, and the
    common points locate only on their boundaries.

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry "geo1" touches geometry "geo2".

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
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Touches(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_Overlaps(geo1, geo2):
    """
    Check whether geometries "spatially overlap". "Spatially overlap" here means two geometries
    intersect but one does not completely contain another.

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry "geo1" overlaps geometry "geo2".

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
      >>> rst = arctern.ST_Overlaps(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
      >>> print(rst)
          0    false
          1    false
          dtype: bool
    """
    import pyarrow as pa
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Overlaps(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_Crosses(geo1, geo2):
    """
    Check whether geometries "spatially cross". "Spatially cross" here means two geometries have
    some, but not all interior points in common. The intersection of the interiors of the geometries
    must not be the empty set and must have a dimensionality less than the maximum dimension of the two
    input geometries.

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry "geo1" crosses geometry "geo2".

    :example:
      >>> import pandas
      >>> import arctern
      >>> data1 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> data2 = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((2 1,3 1,3 2,2 2,2 1))"])
      >>> rst = arctern.ST_Crosses(arctern.ST_GeomFromText(data1), arctern.ST_GeomFromText(data2))
      >>> print(rst)
          0    false
          1    false
          dtype: bool
    """
    import pyarrow as pa
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Crosses(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary')
def ST_IsSimple(geos):
    """
    Check whether geometry is "simple". "Simple" here means that a geometry has no anomalous geometric points
    such as self intersection or self tangency.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry is simple.

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
    return arctern_caller(arctern_core_.ST_IsSimple, arr_geos)

@arctern_udf('binary')
def ST_GeometryType(geos):
    """
    For each geometry in geometries, return a string that indicates is type.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: The type of geometry, e.g., "ST_LINESTRING", "ST_POLYGON", "ST_POINT", "ST_MULTIPOINT".

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))", "POLYGON ((1 1,1 2,2 2,2 1,1 1))"])
      >>> rst = arctern.ST_GeometryType(arctern.ST_GeomFromText(data))
      >>> print(rst)
          0    ST_POLYGON
          1    ST_POLYGON
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    return arctern_caller(arctern_core_.ST_GeometryType, arr_geos)

@arctern_udf('binary')
def ST_MakeValid(geos):
    """
    Create a valid representation of the geometry without losing any of the input vertices. If
    the geometry is already-valid, then nothing will be done.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: Geometry if the input geometry is already-valid or can be made valid. Otherwise, NULL.

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
    return arctern_caller(arctern_core_.ST_MakeValid, arr_geos)

@arctern_udf('binary')
def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    """
    Returns a "simplified" version of the given geometry using the Douglas-Peucker algorithm.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB from.

    :type distance_tolerance: float
    :param distance_tolerance: The maximum distance between a point on a linestring and a curve.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

    :example:
      >>> import pandas
      >>> import arctern
      >>> data = pandas.Series(["POLYGON ((1 1,1 2,2 2,2 1,1 1))","CIRCULARSTRING (0 0,1 1,2 0)"])
      >>> rst = arctern.ST_AsText(arctern.ST_SimplifyPreserveTopology(arctern.ST_GeomFromText(data), 1))
      >>> print(rst)
          0    POLYGON ((1 1,1 2,2 2,2 1,1 1))
          1               LINESTRING (0 0,2 0)
          dtype: object
    """
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    return arctern_caller(arctern_core_.ST_SimplifyPreserveTopology, arr_geos, distance_tolerance)

@arctern_udf('double', 'double', 'double', 'double')
def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    """
    Construct polygon(rectangle) geometries from arr_min_x, arr_min_y, arr_max_x,
    arr_max_y. The edges of polygon are parallel to coordinate axis.

    :type min_x: Series(dtype: float64)
    :param min_x: The minimum value of x coordinate of the rectangles.

    :type min_y: Series(dtype: float64)
    :param min_y: The minimum value of y coordinate of the rectangles.

    :type max_x: Series(dtype: float64)
    :param max_x: The maximum value of x coordinate of the rectangles.

    :type max_y: Series(dtype: float64)
    :param max_y: The maximum value of y coordinate of the rectangles.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

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
    arr_min_x = _to_arrow_array_list(arr_min_x)
    arr_min_y = _to_arrow_array_list(arr_min_y)
    arr_max_x = _to_arrow_array_list(arr_max_x)
    arr_max_y = _to_arrow_array_list(arr_max_y)
    result = arctern_core_.ST_PolygonFromEnvelope(arr_min_x, arr_min_y, arr_max_x, arr_max_y)
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_Contains(geo1, geo2):
    """
    Check whether geometry "geo1" contains geometry "geo2". "geo1 contains geo2" means no points
    of "geo2" lie in the exterior of "geo1" and at least one point of the interior of "geo2" lies
    in the interior of "geo1".

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry "geo1" contains geometry "geo2".

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
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Contains(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_Intersects(geo1, geo2):
    """
    Check whether two geometries intersect (i.e., share any portion of space).

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry "geo1" intersects geometry "geo2".

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
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Intersects(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_Within(geo1, geo2):
    """
    Check whether geometry "geo1" is within geometry "geo2". "geo1 within geo2" means no points of "geo1" lie in the
    exterior of "geo2" and at least one point of the interior of "geo1" lies in the interior of "geo2".

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: bool)
    :return: True if geometry "geo1" within geometry "geo2".

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
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Within(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_Distance(geo1, geo2):
    """
    Calculates the minimum 2D Cartesian (planar) distance between "geo1" and "geo2".

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: float64)
    :return: The value that represents the distance between geometry "geo1" and geometry "geo2".

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
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_Distance(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary', 'binary')
def ST_DistanceSphere(geo1, geo2):
    """
    Returns minimum distance in meters between two lon/lat points.Uses a spherical earth
    and radius derived from the spheroid defined by the SRID.

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: float64)
    :return: The value that represents the distance between geometry "geo1" and geometry "geo2".

    :example:
      >>> import pandas
      >>> import arctern
      >>> p11 = "POINT(10 2)"
      >>> p12 = "POINT(10 2)"
      >>> data1 = pandas.Series([p11, p12])
      >>> p21 = "POINT(10 2)"
      >>> p22 = "POINT(10 3)"
      >>> data2 = pandas.Series([p21, p22])
      >>> rst = arctern.ST_DistanceSphere(arctern.ST_GeomFromText(data2), arctern.ST_GeomFromText(data1))
      >>> print(rst)
          0         1.0
          1    111226.3
          dtype: float64
    """
    import pyarrow as pa
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_DistanceSphere(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary')
def ST_Area(geos):
    """
    Calculate the 2D Cartesian (planar) area of geometry.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: float64)
    :return: The value that represents the area of geometry.

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
    return arctern_caller(arctern_core_.ST_Area, arr_geos)

@arctern_udf('binary')
def ST_Centroid(geos):
    """
    Compute the centroid of geometry.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: The centroid of geometry in WKB form..

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
    return arctern_caller(arctern_core_.ST_Centroid, arr_geos)

@arctern_udf('binary')
def ST_Length(geos):
    """
    Calculate the length of linear geometries.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: float64)
    :return: The value that represents the length of geometry.

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
    return arctern_caller(arctern_core_.ST_Length, arr_geos)

@arctern_udf('binary', 'binary')
def ST_HausdorffDistance(geo1, geo2):
    """
    Returns the Hausdorff distance between two geometries, a measure of how similar
    or dissimilar 2 geometries are.

    :type geo1: Series(dtype: object)
    :param geo1: Geometries in WKB form.

    :type geo2: Series(dtype: object)
    :param geo2: Geometries in WKB form.

    :rtype: Series(dtype: float64)
    :return: The value that represents the hausdorff distance between geo1 and geo2.

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
    arr_geo1 = pa.array(geo1, type='binary')
    arr_geo2 = pa.array(geo2, type='binary')
    arr_geo1 = _to_arrow_array_list(arr_geo1)
    arr_geo2 = _to_arrow_array_list(arr_geo2)
    result = arctern_core_.ST_HausdorffDistance(arr_geo1, arr_geo2)
    return _to_pandas_series(result)

@arctern_udf('binary')
def ST_ConvexHull(geos):
    """
    Compute the smallest convex geometry that encloses all geometries in the given geometry.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: float64)
    :return: Geometries in WKB form.

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
    return arctern_caller(arctern_core_.ST_ConvexHull, arr_geos)

@arctern_udf('binary')
def ST_NPoints(geos):
    """
    Calculates the points number for every geometry in geometries.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: int)
    :return: The number of points.

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
    return arctern_caller(arctern_core_.ST_NPoints, arr_geos)

@arctern_udf('binary')
def ST_Envelope(geos):
    """
    Compute the double-precision minimum bounding box geometry for the given geometry.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

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
    return arctern_caller(arctern_core_.ST_Envelope, arr_geos)

@arctern_udf('binary')
def ST_Buffer(geos, distance):
    """
    Returns a geometry that represents all points whose distance from this geos is
    less than or equal to "distance".

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :type distance: float
    :param distance: The maximum distance of the returned geometry from the given geometry.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

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
    arr_geos = _to_arrow_array_list(arr_geos)
    result = [arctern_core_.ST_Buffer(g, distance) for g in arr_geos]
    return _to_pandas_series(result)

@arctern_udf('binary')
def ST_Union_Aggr(geos):
    """
    Return a geometry that represents the union of a set of geometries.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: Geometry in WKB form.

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
    result = arctern_caller(arctern_core_.ST_Union_Aggr, arr_geos)
    while len(result) > 1:
        result = arctern_caller(arctern_core_.ST_Union_Aggr, result)
    return result

@arctern_udf('binary')
def ST_Envelope_Aggr(geos):
    """
    Compute the double-precision minimum bounding box geometry for the union of given geometries.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: Geometry in WKB form.

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
    result = arctern_caller(arctern_core_.ST_Envelope_Aggr, arr_geos)
    while len(result) > 1:
        result = arctern_caller(arctern_core_.ST_Envelope_Aggr, result)
    return result

@arctern_udf('binary')
def ST_Transform(geos, from_srid, to_srid):
    """
    Return a new geometry with its coordinates transformed from spatial reference system "src" to a "dst".

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :type from_srid: string
    :param from_srid: The current srid of geometries.

    :type to_srid: string
    :param to_srid: The target srid of geometries tranfrom to.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

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
    src = bytes(from_srid, encoding="utf8")
    dst = bytes(to_srid, encoding="utf8")

    return arctern_caller(arctern_core_.ST_Transform, arr_geos, src, dst)

@arctern_udf('binary')
def ST_CurveToLine(geos):
    """
    Convert curves in a geometry to approximate linear representation, e.g., CIRCULAR STRING to regular LINESTRING, CURVEPOLYGON to POLYGON, and
    MULTISURFACE to MULTIPOLYGON. Useful for outputting to devices that can't support
    CIRCULARSTRING geometry types.

    :type geos: Series(dtype: object)
    :param geos: Geometries in WKB form.

    :rtype: Series(dtype: object)
    :return: Geometries in WKB form.

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
    arr_geos = _to_arrow_array_list(arr_geos)
    result = [arctern_core_.ST_CurveToLine(g) for g in arr_geos]
    return _to_pandas_series(result)


def projection(geos, bottom_right, top_left, height, width):
    import pyarrow as pa
    geos = pa.array(geos, type='binary')

    bounding_box_max = bytes(bottom_right, encoding="utf8")
    bounding_box_min = bytes(top_left, encoding="utf8")

    geos_rs = _to_arrow_array_list(geos)

    geos = arctern_core_.projection(geos_rs, bounding_box_max, bounding_box_min, height, width)
    return _to_pandas_series(geos)


def transform_and_projection(geos, src_rs, dst_rs, bottom_right, top_left, height, width):
    import pyarrow as pa
    geos = pa.array(geos, type='binary')

    src = bytes(src_rs, encoding="utf8")
    dst = bytes(dst_rs, encoding="utf8")

    bounding_box_max = bytes(bottom_right, encoding="utf8")
    bounding_box_min = bytes(top_left, encoding="utf8")

    geos_rs = _to_arrow_array_list(geos)

    geos = arctern_core_.transform_and_projection(geos_rs, src, dst, bounding_box_max, bounding_box_min, height, width)
    return _to_pandas_series(geos)


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


def point_map_layer(vega, points, transform=True):
    import pyarrow as pa
    geos = pa.array(points, type='binary')

    # transform and projection handler
    geos_rs = _to_arrow_array_list(geos)

    if transform:
        bounding_box = vega.bounding_box()
        top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
        bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

        height = vega.height()
        width = vega.width()
        coor = vega.coor()

        src = bytes(coor, encoding="utf8")
        dst = bytes('EPSG:3857', encoding="utf8")
        bounding_box_min = bytes(top_left, encoding="utf8")
        bounding_box_max = bytes(bottom_right, encoding="utf8")

        # transform and projection
        if coor != 'EPSG:3857':
            geos_rs = arctern_core_.transform_and_projection(geos_rs, src, dst, bounding_box_max, bounding_box_min, height, width)
        else:
            geos_rs = arctern_core_.projection(geos_rs, bounding_box_max, bounding_box_min, height, width)

    vega_string = vega.build().encode('utf-8')
    rs = arctern_core_.point_map(vega_string, geos_rs)
    return base64.b64encode(rs.buffers()[1].to_pybytes())


# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def weighted_point_map_layer(vega, points, transform=True, **kwargs):
    import pyarrow as pa
    color_weights = kwargs.get('color_weights', None)
    size_weights = kwargs.get('size_weights', None)
    vega_string = vega.build().encode('utf-8')

    geos = pa.array(points, type='binary')

    # transform and projection handler
    geos_rs = _to_arrow_array_list(geos)

    if transform:
        bounding_box = vega.bounding_box()
        top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
        bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

        height = vega.height()
        width = vega.width()
        coor = vega.coor()

        src = bytes(coor, encoding="utf8")
        dst = bytes('EPSG:3857', encoding="utf8")
        bounding_box_min = bytes(top_left, encoding="utf8")
        bounding_box_max = bytes(bottom_right, encoding="utf8")

        # transform and projection
        if coor != 'EPSG:3857':
            geos_rs = arctern_core_.transform_and_projection(geos_rs, src, dst, bounding_box_max, bounding_box_min, height, width)
        else:
            geos_rs = arctern_core_.projection(geos_rs, bounding_box_max, bounding_box_min, height, width)

    if color_weights is None and size_weights is None:
        rs = arctern_core_.weighted_point_map(vega_string, geos_rs)
    elif color_weights is not None and size_weights is not None:
        if color_weights.dtypes == 'float64':
            arr_c = pa.array(color_weights, type='double')
        else:
            arr_c = pa.array(color_weights, type='int64')
        if size_weights.dtypes == 'float64':
            arr_s = pa.array(size_weights, type='double')
        else:
            arr_s = pa.array(size_weights, type='int64')
        color_weights_rs = _to_arrow_array_list(arr_c)
        size_weights_rs = _to_arrow_array_list(arr_s)
        rs = arctern_core_.weighted_color_size_point_map(vega_string, geos_rs, color_weights_rs, size_weights_rs)
    elif color_weights is None and size_weights is not None:
        if size_weights.dtypes == 'float64':
            arr_s = pa.array(size_weights, type='double')
        else:
            arr_s = pa.array(size_weights, type='int64')
        size_weights_rs = _to_arrow_array_list(arr_s)
        rs = arctern_core_.weighted_size_point_map(vega_string, geos_rs, size_weights_rs)
    else:
        if color_weights.dtypes == 'float64':
            arr_c = pa.array(color_weights, type='double')
        else:
            arr_c = pa.array(color_weights, type='int64')
        color_weights_rs = _to_arrow_array_list(arr_c)
        rs = arctern_core_.weighted_color_point_map(vega_string, geos_rs, color_weights_rs)

    return base64.b64encode(rs.buffers()[1].to_pybytes())


def heat_map_layer(vega, points, weights, transform=True):
    import pyarrow as pa
    geos = pa.array(points, type='binary')

    # transform and projection handler
    geos_rs = _to_arrow_array_list(geos)

    if transform:
        bounding_box = vega.bounding_box()
        top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
        bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

        height = vega.height()
        width = vega.width()
        coor = vega.coor()

        src = bytes(coor, encoding="utf8")
        dst = bytes('EPSG:3857', encoding="utf8")
        bounding_box_min = bytes(top_left, encoding="utf8")
        bounding_box_max = bytes(bottom_right, encoding="utf8")

        # transform and projection
        if coor != 'EPSG:3857':
            geos_rs = arctern_core_.transform_and_projection(geos_rs, src, dst, bounding_box_max, bounding_box_min, height, width)
        else:
            geos_rs = arctern_core_.projection(geos_rs, bounding_box_max, bounding_box_min, height, width)

    # weights handler
    if weights.dtypes == 'float64':
        arr = pa.array(weights, type='double')
    else:
        arr = pa.array(weights, type='int64')

    weights_rs = _to_arrow_array_list(arr)

    vega_string = vega.build().encode('utf-8')
    rs = arctern_core_.heat_map(vega_string, geos_rs, weights_rs)
    return base64.b64encode(rs.buffers()[1].to_pybytes())


def choropleth_map_layer(vega, region_boundaries, weights, transform=True):
    import pyarrow as pa
    geos = pa.array(region_boundaries, type='binary')

    # transform and projection handler
    geos_rs = _to_arrow_array_list(geos)

    if transform:
        bounding_box = vega.bounding_box()
        top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
        bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

        height = vega.height()
        width = vega.width()
        coor = vega.coor()

        src = bytes(coor, encoding="utf8")
        dst = bytes('EPSG:3857', encoding="utf8")
        bounding_box_min = bytes(top_left, encoding="utf8")
        bounding_box_max = bytes(bottom_right, encoding="utf8")

        # transform and projection
        if coor != 'EPSG:3857':
            geos_rs = arctern_core_.transform_and_projection(geos_rs, src, dst, bounding_box_max, bounding_box_min, height, width)
        else:
            geos_rs = arctern_core_.projection(geos_rs, bounding_box_max, bounding_box_min, height, width)

    vega_string = vega.build().encode('utf-8')

    # weights handler
    if weights.dtypes == 'float64':
        arr = pa.array(weights, type='double')
    else:
        arr = pa.array(weights, type='int64')

    weights_rs = _to_arrow_array_list(arr)

    rs = arctern_core_.choropleth_map(vega_string, geos_rs, weights_rs)
    return base64.b64encode(rs.buffers()[1].to_pybytes())


def icon_viz_layer(vega, points, transform=True):
    import pyarrow as pa
    geos = pa.array(points, type='binary')

    # transform and projection handler
    geos_rs = _to_arrow_array_list(geos)

    if transform:
        bounding_box = vega.bounding_box()
        top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
        bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

        height = vega.height()
        width = vega.width()
        coor = vega.coor()

        src = bytes(coor, encoding="utf8")
        dst = bytes('EPSG:3857', encoding="utf8")
        bounding_box_min = bytes(top_left, encoding="utf8")
        bounding_box_max = bytes(bottom_right, encoding="utf8")

        # transform and projection
        if coor != 'EPSG:3857':
            geos_rs = arctern_core_.transform_and_projection(geos_rs, src, dst, bounding_box_max, bounding_box_min, height, width)
        else:
            geos_rs = arctern_core_.projection(geos_rs, bounding_box_max, bounding_box_min, height, width)

    vega_string = vega.build().encode('utf-8')

    rs = arctern_core_.icon_viz(vega_string, geos_rs)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def fishnet_map_layer(vega, points, weights, transform=True):
    import pyarrow as pa
    geos = pa.array(points, type='binary')

    # transform and projection handler
    geos_rs = _to_arrow_array_list(geos)

    if transform:
        bounding_box = vega.bounding_box()
        top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
        bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

        height = vega.height()
        width = vega.width()
        coor = vega.coor()

        src = bytes(coor, encoding="utf8")
        dst = bytes('EPSG:3857', encoding="utf8")
        bounding_box_min = bytes(top_left, encoding="utf8")
        bounding_box_max = bytes(bottom_right, encoding="utf8")

        # transform and projection
        if coor != 'EPSG:3857':
            geos_rs = arctern_core_.transform_and_projection(geos_rs, src, dst, bounding_box_max, bounding_box_min, height, width)
        else:
            geos_rs = arctern_core_.projection(geos_rs, bounding_box_max, bounding_box_min, height, width)

    # weights handler
    if weights.dtypes == 'float64':
        arr = pa.array(weights, type='double')
    else:
        arr = pa.array(weights, type='int64')

    weights_rs = _to_arrow_array_list(arr)

    vega_string = vega.build().encode('utf-8')
    rs = arctern_core_.fishnet_map(vega_string, geos_rs, weights_rs)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def version():
    """
    :return: version of arctern
    """
    return arctern_core_.GIS_Version().decode("utf-8")
