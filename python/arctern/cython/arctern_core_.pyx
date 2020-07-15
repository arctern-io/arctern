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

# cython: language_level=3
# distutils: language = c++

from pyarrow.lib cimport (shared_ptr, CArray, CChunkedArray, pyarrow_wrap_array, pyarrow_unwrap_array, pyarrow_wrap_chunked_array, pyarrow_unwrap_chunked_array)
from libcpp.vector cimport vector
from libcpp.string cimport string
cimport arctern_core__ as arctern_core_pxd


# render func api:
def projection(geos_list, bottom_right, top_left, height, width):
    cdef vector[shared_ptr[CArray]] geos_vector
    for geos in geos_list:
        arr = pyarrow_unwrap_array(geos)
        geos_vector.push_back(arr)
    cdef vector[shared_ptr[CArray]] output_geos
    result = arctern_core_pxd.projection(geos_vector, bottom_right, top_left, height, width)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def transform_and_projection(geos_list, src_rs, dst_rs, bottom_right, top_left, height, width):
    cdef vector[shared_ptr[CArray]] geos_vector
    for geos in geos_list:
        arr = pyarrow_unwrap_array(geos)
        geos_vector.push_back(arr)
    cdef vector[shared_ptr[CArray]] output_geos
    result = arctern_core_pxd.transform_and_projection(geos_vector, src_rs, dst_rs, bottom_right, top_left, height, width)
    return [pyarrow_wrap_array(ptr) for ptr in result]

# render drawing api:
def point_map(vega, points_list):
    cdef vector[shared_ptr[CArray]] points_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.point_map(points_vector, vega))

def weighted_point_map(vega, points_list):
    cdef vector[shared_ptr[CArray]] points_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.weighted_point_map(points_vector, vega))

def weighted_color_point_map(vega, points_list, color_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] color_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for color in color_list:
        arr = pyarrow_unwrap_array(color)
        color_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.weighted_point_map(points_vector, color_vector, vega))

def weighted_size_point_map(vega, points_list, size_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] size_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for size in size_list:
        arr = pyarrow_unwrap_array(size)
        size_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.weighted_point_map(points_vector, size_vector, vega))

def weighted_color_size_point_map(vega, points_list, color_list, size_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] color_vector
    cdef vector[shared_ptr[CArray]] size_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for color in color_list:
        arr = pyarrow_unwrap_array(color)
        color_vector.push_back(arr)
    for size in size_list:
        arr = pyarrow_unwrap_array(size)
        size_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.weighted_point_map(points_vector, color_vector, size_vector, vega))

def heat_map(vega, points_list, weights_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] weights_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for weights in weights_list:
        arr = pyarrow_unwrap_array(weights)
        weights_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.heat_map(points_vector, weights_vector, vega))

def choropleth_map(vega,region_boundaries_list, weights_list):
    cdef vector[shared_ptr[CArray]] region_boundaries_vector
    cdef vector[shared_ptr[CArray]] weights_vector
    for region in region_boundaries_list:
        arr = pyarrow_unwrap_array(region)
        region_boundaries_vector.push_back(arr)
    for weights in weights_list:
        arr = pyarrow_unwrap_array(weights)
        weights_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.choropleth_map(region_boundaries_vector, weights_vector, vega))

def unique_value_choropleth_map(vega,region_boundaries_list, labels_list):
    cdef vector[shared_ptr[CArray]] region_boundaries_vector
    cdef vector[shared_ptr[CArray]] labels_vector
    for region in region_boundaries_list:
        arr = pyarrow_unwrap_array(region)
        region_boundaries_vector.push_back(arr)
    for labels in labels_list:
        arr = pyarrow_unwrap_array(labels)
        labels_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.unique_value_choropleth_map(region_boundaries_vector, labels_vector, vega))

def icon_viz(vega, points_list):
    cdef vector[shared_ptr[CArray]] points_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)

    return pyarrow_wrap_array(arctern_core_pxd.icon_viz(points_vector, vega))

def fishnet_map(vega, points_list, weights_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] weights_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for weights in weights_list:
        arr = pyarrow_unwrap_array(weights)
        weights_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.fishnet_map(points_vector, weights_vector, vega))

# gis api:
def ST_Point(object arr_x,object arr_y):
    cdef vector[shared_ptr[CArray]] points_x
    for arr in arr_x:
        points_x.push_back(pyarrow_unwrap_array(arr))
    
    cdef vector[shared_ptr[CArray]] points_y
    for arr in arr_y:
        points_y.push_back(pyarrow_unwrap_array(arr))

    points = arctern_core_pxd.ST_Point(points_x, points_y)
    return [pyarrow_wrap_array(ptr) for ptr in points]

def ST_GeomFromGeoJSON(object json):
    result = arctern_core_pxd.ST_GeomFromGeoJSON(pyarrow_unwrap_array(json))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_GeomFromText(object text):
    result = arctern_core_pxd.ST_GeomFromText(pyarrow_unwrap_array(text))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_AsText(object text):
    result = arctern_core_pxd.ST_AsText(pyarrow_unwrap_array(text))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_AsGeoJSON(object text):
    result = arctern_core_pxd.ST_AsGeoJSON(pyarrow_unwrap_array(text))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Intersection(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Intersection(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_IsValid(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_IsValid(pyarrow_unwrap_array(geometries)))

def ST_Equals(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Equals(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Disjoint(object left_geometries,object right_geometries ):
    result = arctern_core_pxd.ST_Disjoint(pyarrow_unwrap_chunked_array(left_geometries),pyarrow_unwrap_chunked_array(right_geometries))
    return pyarrow_wrap_chunked_array(result)

def ST_Union(object left_geometries,object right_geometries ):
    result = arctern_core_pxd.ST_Union(pyarrow_unwrap_chunked_array(left_geometries),pyarrow_unwrap_chunked_array(right_geometries))
    return pyarrow_wrap_chunked_array(result)

def ST_Boundary(object geometries ):
    result = arctern_core_pxd.ST_Boundary(pyarrow_unwrap_chunked_array(geometries))
    return pyarrow_wrap_chunked_array(result)

def ST_Touches(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Touches(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Overlaps(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Overlaps(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Crosses(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Crosses(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_IsSimple(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_IsSimple(pyarrow_unwrap_array(geometries)))

def ST_GeometryType(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_GeometryType(pyarrow_unwrap_array(geometries)))

def ST_MakeValid(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_MakeValid(pyarrow_unwrap_array(geometries)))

def ST_PrecisionReduce(object geometries,int num_dat):
    return pyarrow_wrap_array(arctern_core_pxd.ST_PrecisionReduce(pyarrow_unwrap_array(geometries),num_dat))

def ST_Translate(object geometries, double shifter_x, double shifter_y):
    return pyarrow_wrap_chunked_array(arctern_core_pxd.ST_Translate(pyarrow_unwrap_chunked_array(geometries), shifter_x, shifter_y))

def ST_Rotate(object geometries, double rotation_angle, double origin_x, double origin_y):
    return pyarrow_wrap_chunked_array(arctern_core_pxd.ST_Rotate(pyarrow_unwrap_chunked_array(geometries), rotation_angle, origin_x, origin_y))

def ST_Rotate2(object geometries, double rotation_angle, string origin):
    cdef string origin_ = origin
    return pyarrow_wrap_chunked_array(arctern_core_pxd.ST_Rotate(pyarrow_unwrap_chunked_array(geometries), rotation_angle, origin_))

def ST_SimplifyPreserveTopology(object geometries,double distanceTolerance):
    return pyarrow_wrap_array(arctern_core_pxd.ST_SimplifyPreserveTopology(pyarrow_unwrap_array(geometries),distanceTolerance))

def ST_PolygonFromEnvelope(object min_x,object min_y,object max_x,object max_y):
    cdef vector[shared_ptr[CArray]] min_x_arr
    for arr in min_x:
        min_x_arr.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] min_y_arr
    for arr in min_y:
        min_y_arr.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] max_x_arr
    for arr in max_x:
        max_x_arr.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] max_y_arr
    for arr in max_y:
        max_y_arr.push_back(pyarrow_unwrap_array(arr))

    result = arctern_core_pxd.ST_PolygonFromEnvelope(min_x_arr, min_y_arr, max_x_arr, max_y_arr)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Contains(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Contains(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Intersects(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Intersects(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Within(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Within(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Within2(left_geometries, right_geometry):
    cdef string s = right_geometry
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Within(left, s)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Distance(object geo_arr1,object geo_arr2):
    cdef vector[shared_ptr[CArray]] arr_1
    for arr in geo_arr1:
        arr_1.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] arr_2
    for arr in geo_arr2:
        arr_2.push_back(pyarrow_unwrap_array(arr))
    result = arctern_core_pxd.ST_Distance(arr_1, arr_2)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_DistanceSphere(object geo_arr1,object geo_arr2):
    cdef vector[shared_ptr[CArray]] arr_1
    for arr in geo_arr1:
        arr_1.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] arr_2
    for arr in geo_arr2:
        arr_2.push_back(pyarrow_unwrap_array(arr))
    result = arctern_core_pxd.ST_DistanceSphere(arr_1, arr_2)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Area(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Area(pyarrow_unwrap_array(geo_arr)))

def ST_Centroid(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Centroid(pyarrow_unwrap_array(geo_arr)))

def ST_Length(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Length(pyarrow_unwrap_array(geo_arr)))

def ST_HausdorffDistance(object geo1, object geo2):
    cdef vector[shared_ptr[CArray]] arr_1
    for arr in geo1:
        arr_1.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] arr_2
    for arr in geo2:
        arr_2.push_back(pyarrow_unwrap_array(arr))
    result = arctern_core_pxd.ST_HausdorffDistance(arr_1, arr_2)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_ConvexHull(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_ConvexHull(pyarrow_unwrap_array(geo_arr)))

def ST_Transform(object geo_arr, bytes src_rs, bytes dst_rs):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Transform(pyarrow_unwrap_array(geo_arr),src_rs,dst_rs))

def ST_CurveToLine(object geo_arr):
    result = arctern_core_pxd.ST_CurveToLine(pyarrow_unwrap_array(geo_arr))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_SymDifference(object geo1,object geo2):
    result = arctern_core_pxd.ST_SymDifference(pyarrow_unwrap_chunked_array(geo1),pyarrow_unwrap_chunked_array(geo2))
    return pyarrow_wrap_chunked_array(result)

def ST_Difference(object geo1,object geo2):
    result = arctern_core_pxd.ST_Difference(pyarrow_unwrap_chunked_array(geo1),pyarrow_unwrap_chunked_array(geo2))
    return pyarrow_wrap_chunked_array(result)

def ST_ExteriorRing(object geos):
    result = arctern_core_pxd.ST_ExteriorRing(pyarrow_unwrap_chunked_array(geos))
    return pyarrow_wrap_chunked_array(result)

def ST_IsEmpty(object geos):
    result = arctern_core_pxd.ST_IsEmpty(pyarrow_unwrap_chunked_array(geos))
    return pyarrow_wrap_chunked_array(result)

def ST_Scale2(object geos, double factor_x, double factor_y, string origin):
    cdef string origin_ = origin
    result = arctern_core_pxd.ST_Scale(pyarrow_unwrap_chunked_array(geos), factor_x, factor_y, origin_)
    return pyarrow_wrap_chunked_array(result)

def ST_Scale(object geos, double factor_x, double factor_y, double origin_x, double origin_y):
    result = arctern_core_pxd.ST_Scale(pyarrow_unwrap_chunked_array(geos), factor_x, factor_y, origin_x, origin_y)
    return pyarrow_wrap_chunked_array(result)

def ST_Affine(object geos, double a, double b, double d, double e, double offset_x, double offset_y):
    result = arctern_core_pxd.ST_Affine(pyarrow_unwrap_chunked_array(geos), a, b, d, e, offset_x, offset_y)
    return pyarrow_wrap_chunked_array(result)

def ST_NPoints(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_NPoints(pyarrow_unwrap_array(geo_arr)))

def ST_Envelope(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Envelope(pyarrow_unwrap_array(geo_arr)))

def ST_Buffer(object geo_arr,double dfDist, n_quadrant_segments = None):
    if n_quadrant_segments is not None:
        result = arctern_core_pxd.ST_Buffer(pyarrow_unwrap_array(geo_arr),dfDist,n_quadrant_segments)
        return [pyarrow_wrap_array(ptr) for ptr in result]
    else:
        result = arctern_core_pxd.ST_Buffer(pyarrow_unwrap_array(geo_arr),dfDist)
        return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_IndexedWithin(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_IndexedWithin(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Union_Aggr(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Union_Aggr(pyarrow_unwrap_array(geo_arr)))

def ST_Envelope_Aggr(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Envelope_Aggr(pyarrow_unwrap_array(geo_arr)))

def GIS_Version():
    return arctern_core_pxd.GIS_Version()

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

cdef class SpatialIndex:
    cdef arctern_core_pxd.GeosIndex* thisptr
    cdef object data

    def __cinit__(self):
        self.thisptr = new arctern_core_pxd.GeosIndex()

    def __dealloc__(self):
        del self.thisptr

    def Append(self, geometries):
        self.data = geometries
        import pyarrow as pa
        arr_geos = pa.array(geometries, type='binary')
        list_geos = _to_arrow_array_list(arr_geos)
        cdef vector[shared_ptr[CArray]] geos
        for geo in list_geos:
            geos.push_back(pyarrow_unwrap_array(geo))
        self.thisptr.append(geos)

    def near_road(self, gps_points, distance):
        import pyarrow as pa
        arr_geos = pa.array(gps_points, type='binary')
        list_gps_points = _to_arrow_array_list(arr_geos)
        cdef vector[shared_ptr[CArray]] gps_points_to_match
        for gps_point in list_gps_points:
            gps_points_to_match.push_back(pyarrow_unwrap_array(gps_point))
        result = self.thisptr.near_road(gps_points_to_match, distance)
        pyarrow_res = [pyarrow_wrap_array(ptr) for ptr in result]
        return _to_pandas_series(pyarrow_res)

    def nearest_location_on_road(self, gps_points):
        import pyarrow as pa
        arr_geos = pa.array(gps_points, type='binary')
        list_gps_points = _to_arrow_array_list(arr_geos)
        cdef vector[shared_ptr[CArray]] gps_points_to_match
        for gps_point in list_gps_points:
            gps_points_to_match.push_back(pyarrow_unwrap_array(gps_point))
        result = self.thisptr.nearest_location_on_road(gps_points_to_match)
        pyarrow_res = [pyarrow_wrap_array(ptr) for ptr in result]
        return _to_pandas_series(pyarrow_res)

    def nearest_road(self, gps_points):
        import pyarrow as pa
        arr_geos = pa.array(gps_points, type='binary')
        list_gps_points = _to_arrow_array_list(arr_geos)
        cdef vector[shared_ptr[CArray]] gps_points_to_match
        for gps_point in list_gps_points:
            gps_points_to_match.push_back(pyarrow_unwrap_array(gps_point))
        result = self.thisptr.nearest_road(gps_points_to_match)
        pyarrow_res = [pyarrow_wrap_array(ptr) for ptr in result]
        return _to_pandas_series(pyarrow_res)

    def within_which(self, left):
        import pyarrow as pa
        import pandas
        arr_left = pa.array(left, type='binary')
        list_left = _to_arrow_array_list(arr_left)
        cdef vector[shared_ptr[CArray]] left_to_match
        for geo in list_left:
            left_to_match.push_back(pyarrow_unwrap_array(geo))
        result = self.thisptr.ST_IndexedWithin(left_to_match)
        pyarrow_res = [pyarrow_wrap_array(ptr) for ptr in result]
        res = _to_pandas_series(pyarrow_res)
        res = res.apply(lambda x: self.data.index[x] if x >= 0 else pandas.NA)
        res = res.set_axis(left.index)
        return res

    def query(self, inputs):
        import pyarrow as pa
        arr_geos = pa.array(inputs, type='binary')
        list_geos = _to_arrow_array_list(arr_geos)
        cdef vector[shared_ptr[CArray]] geos_to_match
        for geo in list_geos:
            geos_to_match.push_back(pyarrow_unwrap_array(geo))
        result = self.thisptr.query(geos_to_match)
        pyarrow_res = [pyarrow_wrap_array(ptr) for ptr in result]
        return _to_pandas_series(pyarrow_res)
