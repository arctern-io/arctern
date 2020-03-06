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

from pyarrow.lib cimport *
from libcpp.string cimport *


cdef extern from "render.h" namespace "zilliz::render":
    shared_ptr[CArray] point_map(const shared_ptr[CArray] &ptr_x,const shared_ptr[CArray] &ptr_y,const string &conf)
    shared_ptr[CArray] heat_map(const shared_ptr[CArray] &ptr_x,const shared_ptr[CArray] &ptr_y,const shared_ptr[CArray] &ptr_c,const string &conf)
    shared_ptr[CArray] choropleth_map(const shared_ptr[CArray] &ptr_wkt,const shared_ptr[CArray] &ptr_count,const string &conf)


cdef extern from "gis.h" namespace "zilliz::gis":
    shared_ptr[CArray] ST_Point(const shared_ptr[CArray] &ptr_x,const shared_ptr[CArray] &ptr_y)
    shared_ptr[CArray] ST_GeomFromGeoJSON(const shared_ptr[CArray] &json)
    shared_ptr[CArray] ST_Intersection(shared_ptr[CArray] &left_geometries,shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_IsValid(const shared_ptr[CArray] &geometries)
    shared_ptr[CArray] ST_Equals(const shared_ptr[CArray] &left_geometries, const shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_Touches(const shared_ptr[CArray] &left_geometries, const shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_Overlaps(const shared_ptr[CArray] &left_geometries,const shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_Crosses(const shared_ptr[CArray] &left_geometries, const shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_IsSimple(const shared_ptr[CArray] &geometries)
    shared_ptr[CArray] ST_PrecisionReduce(const shared_ptr[CArray] &geometries, int32_t num_dot)
    shared_ptr[CArray] ST_GeometryType(const shared_ptr[CArray] &geometries)
    shared_ptr[CArray] ST_MakeValid(const shared_ptr[CArray] &geometries)
    shared_ptr[CArray] ST_SimplifyPreserveTopology(const shared_ptr[CArray] &geometries, double distanceTolerance)
    shared_ptr[CArray] ST_PolygonFromEnvelope(const shared_ptr[CArray] &min_x,const shared_ptr[CArray] &min_y,const shared_ptr[CArray] &max_x,const shared_ptr[CArray] &max_y)
    shared_ptr[CArray] ST_Contains(const shared_ptr[CArray] &ptr_x,const shared_ptr[CArray] &ptr_y)
    shared_ptr[CArray] ST_Intersects(const shared_ptr[CArray] &geo_arr1,const shared_ptr[CArray] &geo_arr2)
    shared_ptr[CArray] ST_Within(const shared_ptr[CArray] &geo_arr1,const shared_ptr[CArray] &geo_arr2)
    shared_ptr[CArray] ST_Distance(const shared_ptr[CArray] &geo_arr1,const shared_ptr[CArray] &geo_arr2)
    shared_ptr[CArray] ST_Area(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Centroid(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Length(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_HausdorffDistance(const shared_ptr[CArray] &geo1,const shared_ptr[CArray] &geo2)
    shared_ptr[CArray] ST_ConvexHull(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Transform(const shared_ptr[CArray] &geo_arr, const string& src_rs, const string& dst_rs)
    shared_ptr[CArray] ST_CurveToLine(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_NPoints(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Envelope(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Buffer(const shared_ptr[CArray] &geo_arr, double dfDist)
    shared_ptr[CArray] ST_Buffer(const shared_ptr[CArray] &geo_arr, double dfDist, int n_quadrant_segments)
    shared_ptr[CArray] ST_Union_Aggr(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Envelope_Aggr(const shared_ptr[CArray] &geo_arr)
