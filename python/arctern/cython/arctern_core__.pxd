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

from pyarrow.lib cimport (shared_ptr, CArray, int32_t)
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "render.h" namespace "arctern::render":
    # func api:
    const vector[shared_ptr[CArray]] projection(const vector[shared_ptr[CArray]] &geos,const string &bottom_right,const string &top_left,const int &height,const int &width) except +
    const vector[shared_ptr[CArray]] transform_and_projection(const vector[shared_ptr[CArray]] &geos,const string &src_rs,const string &dst_rs,const string &bottom_right,const string &top_left,const int &height,const int &width) except +

    # drawing api:
    shared_ptr[CArray] point_map(const vector[shared_ptr[CArray]] &points_vector, const string &vega) except +
    shared_ptr[CArray] weighted_point_map(const vector[shared_ptr[CArray]] &points_vector, const string &vega) except +
    shared_ptr[CArray] weighted_point_map(const vector[shared_ptr[CArray]] &points_vector, const vector[shared_ptr[CArray]] &weights_vector, const string &vega) except +
    shared_ptr[CArray] weighted_point_map(const vector[shared_ptr[CArray]] &points_vector, const vector[shared_ptr[CArray]] &color_weights_vector, const vector[shared_ptr[CArray]] &size_weights_vector, const string &vega) except +
    shared_ptr[CArray] heat_map(const vector[shared_ptr[CArray]] &points_vector, const vector[shared_ptr[CArray]] &weights_vector, const string &vega) except +
    shared_ptr[CArray] choropleth_map(const vector[shared_ptr[CArray]] &region_boundaries_vector, const vector[shared_ptr[CArray]] &weights_vector, const string &vega) except +
    shared_ptr[CArray] icon_viz(const vector[shared_ptr[CArray]] &points_vector, const string &vega) except +
    shared_ptr[CArray] fishnet_map(const vector[shared_ptr[CArray]] &points_vector, const vector[shared_ptr[CArray]] &weights_vector, const string &vega) except +

cdef extern from "gis.h" namespace "arctern::gis":
    vector[shared_ptr[CArray]] ST_Point(const vector[shared_ptr[CArray]] &ptr_x, \
                                        const vector[shared_ptr[CArray]] &ptr_y) except +
    vector[shared_ptr[CArray]] ST_GeomFromGeoJSON(const shared_ptr[CArray] &json) except +
    vector[shared_ptr[CArray]] ST_GeomFromText(const shared_ptr[CArray] &text) except +
    vector[shared_ptr[CArray]] ST_AsText(const shared_ptr[CArray] &text) except +
    vector[shared_ptr[CArray]] ST_AsGeoJSON(const shared_ptr[CArray] &text) except +

    vector[shared_ptr[CArray]] ST_Intersection(const vector[shared_ptr[CArray]] &left_geometries, \
                                               const vector[shared_ptr[CArray]] &right_geometries) except +

    shared_ptr[CArray] ST_IsValid(const shared_ptr[CArray] &geometries) except +

    vector[shared_ptr[CArray]] ST_Equals(const vector[shared_ptr[CArray]] &left_geometries, \
                                         const vector[shared_ptr[CArray]] &right_geometries) except +

    vector[shared_ptr[CArray]] ST_Touches(const vector[shared_ptr[CArray]] &left_geometries, \
                                          const vector[shared_ptr[CArray]] &right_geometries) except +

    vector[shared_ptr[CArray]] ST_Overlaps(const vector[shared_ptr[CArray]] &left_geometries, \
                                           const vector[shared_ptr[CArray]] &right_geometries) except +

    vector[shared_ptr[CArray]] ST_Crosses(const vector[shared_ptr[CArray]] &left_geometries, \
                                          const vector[shared_ptr[CArray]] &right_geometries) except +

    shared_ptr[CArray] ST_IsSimple(const shared_ptr[CArray] &geometries) except +
    shared_ptr[CArray] ST_PrecisionReduce(const shared_ptr[CArray] &geometries, int32_t num_dot) except +
    shared_ptr[CArray] ST_GeometryType(const shared_ptr[CArray] &geometries) except +
    shared_ptr[CArray] ST_MakeValid(const shared_ptr[CArray] &geometries) except +
    shared_ptr[CArray] ST_SimplifyPreserveTopology(const shared_ptr[CArray] &geometries, double distanceTolerance) except +

    vector[shared_ptr[CArray]] ST_PolygonFromEnvelope(const vector[shared_ptr[CArray]] &min_x, \
                                                      const vector[shared_ptr[CArray]] &min_y, \
                                                      const vector[shared_ptr[CArray]] &max_x, \
                                                      const vector[shared_ptr[CArray]] &max_y) except +
    
    vector[shared_ptr[CArray]] ST_Contains(const vector[shared_ptr[CArray]] &ptr_x, \
                                           const vector[shared_ptr[CArray]] &ptr_y) except +

    vector[shared_ptr[CArray]] ST_Intersects(const vector[shared_ptr[CArray]] &geo_arr1, \
                                             const vector[shared_ptr[CArray]] &geo_arr2) except +

    vector[shared_ptr[CArray]] ST_Within(const vector[shared_ptr[CArray]] &geo_arr1, \
                                         const vector[shared_ptr[CArray]] &geo_arr2) except +

    vector[shared_ptr[CArray]] ST_Within(const vector[shared_ptr[CArray]] &geo_arr1, \
                                         const string& geo2) except +

    vector[shared_ptr[CArray]] ST_Distance(const vector[shared_ptr[CArray]] &geo_arr1, \
                                           const vector[shared_ptr[CArray]] &geo_arr2) except +

    vector[shared_ptr[CArray]] ST_DistanceSphere(const vector[shared_ptr[CArray]] &geo_arr1, \
                                                 const vector[shared_ptr[CArray]] &geo_arr2) except +

    shared_ptr[CArray] ST_Area(const shared_ptr[CArray] &geo_arr) except +
    shared_ptr[CArray] ST_Centroid(const shared_ptr[CArray] &geo_arr) except +
    shared_ptr[CArray] ST_Length(const shared_ptr[CArray] &geo_arr) except +

    vector[shared_ptr[CArray]] ST_HausdorffDistance(vector[shared_ptr[CArray]] &geo1, \
                                                    vector[shared_ptr[CArray]] &geo2) except +

    shared_ptr[CArray] ST_ConvexHull(const shared_ptr[CArray] &geo_arr) except +
    shared_ptr[CArray] ST_Transform(const shared_ptr[CArray] &geo_arr, const string& src_rs, const string& dst_rs) except +

    vector[shared_ptr[CArray]] ST_CurveToLine(const shared_ptr[CArray] &geo_arr) except +

    shared_ptr[CArray] ST_NPoints(const shared_ptr[CArray] &geo_arr) except +
    shared_ptr[CArray] ST_Envelope(const shared_ptr[CArray] &geo_arr) except +

    vector[shared_ptr[CArray]] ST_Buffer(const shared_ptr[CArray] &geo_arr, double dfDist) except +
    vector[shared_ptr[CArray]] ST_Buffer(const shared_ptr[CArray] &geo_arr, double dfDist, int n_quadrant_segments) except +

    shared_ptr[CArray] ST_Union_Aggr(const shared_ptr[CArray] &geo_arr) except +
    shared_ptr[CArray] ST_Envelope_Aggr(const shared_ptr[CArray] &geo_arr) except +

    vector[shared_ptr[CArray]] ST_IndexedWithin(const vector[shared_ptr[CArray]] &points_raw, const vector[shared_ptr[CArray]] &polygons_raw) except +

    string GIS_Version() except +

cdef extern from "map_match.h" namespace "arctern::map_match":
    vector[shared_ptr[CArray]] nearest_location_on_road(const vector[shared_ptr[CArray]] &roads, \
                                                        const vector[shared_ptr[CArray]] &gps_points) except +
    vector[shared_ptr[CArray]] nearest_road(const vector[shared_ptr[CArray]] &roads, \
                                            const vector[shared_ptr[CArray]] &gps_points) except +
    vector[shared_ptr[CArray]] near_road(const vector[shared_ptr[CArray]] &roads, \
                                            const vector[shared_ptr[CArray]] &gps_points, const double distance) except +
