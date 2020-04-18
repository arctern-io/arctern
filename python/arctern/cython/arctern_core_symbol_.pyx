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

def projection(geos, bottom_right, top_left, int height, int width):
    pass

def transform_and_projection(geos, src_rs, dst_rs, bottom_right, top_left, int height, int width):
    pass

def point_map_wkb(points, conf):
    pass

def weighted_point_map_wkb_0_0(points, conf):
    pass

def weighted_point_map_wkb_1_0(points, conf, cs):
    pass

def weighted_point_map_wkb_0_1(points, conf, ss):
    pass

def weighted_point_map_wkb_1_1(points, conf, cs, ss):
    pass

def weighted_point_map_0_0(arr_x, arr_y, conf):
    pass

def weighted_point_map_0_1(arr_x, arr_y, conf, ss):
    pass

def weighted_point_map_1_0(arr_x, arr_y, conf, cs):
    pass

def weighted_point_map_1_1(arr_x, arr_y, conf, cs, ss):
    pass

def heat_map_wkb(points, arr_c, conf):
    pass

def point_map(arr_x, arr_y, conf):
    pass

def heat_map(arr_x, arr_y, arr_c, conf):
    pass

def choropleth_map(arr_wkt, arr_count, conf):
    pass

def icon_viz(points, conf):
    pass

def wkt2wkb(arr_wkt):
    pass

def wkb2wkt(arr_wkb):
    pass

def ST_Point(object arr_x,object arr_y):
    pass

def ST_GeomFromGeoJSON(object json):
    pass

def ST_GeomFromText(object text):
    pass

def ST_Intersection(object left_geometries,object right_geometries):
    pass

def ST_IsValid(object geometries):
    pass

def ST_Equals(object left_geometries,object right_geometries):
    pass

def ST_Touches(object left_geometries,object right_geometries):
    pass

def ST_Overlaps(object left_geometries,object right_geometries):
    pass

def ST_Crosses(object left_geometries,object right_geometries):
    pass

def ST_IsSimple(object geometries):
    pass

def ST_GeometryType(object geometries):
    pass

def ST_MakeValid(object geometries):
    pass

def ST_PrecisionReduce(object geometries,int num_dat):
    pass

def ST_SimplifyPreserveTopology(object geometries,double distanceTolerance):
    pass

def ST_PolygonFromEnvelope(object min_x,object min_y,object max_x,object max_y):
    pass

def ST_Contains(object ptr_x,object ptr_y):
    pass

def ST_Intersects(object geo_arr1,object geo_arr2):
    pass

def ST_Within(object geo_arr1,object geo_arr2):
    pass

def ST_Distance(object geo_arr1,object geo_arr2):
    pass

def ST_DistanceSphere(object geo_arr1,object geo_arr2):
    pass

def ST_Area(object geo_arr):
    pass

def ST_Centroid(object geo_arr):
    pass

def ST_Length(object geo_arr):
    pass

def ST_HausdorffDistance(object geo1, object geo2):
    pass

def ST_ConvexHull(object geo_arr):
    pass

def ST_Transform(object geo_arr, bytes src_rs, bytes dst_rs):
    pass

def ST_CurveToLine(object geo_arr):
    pass

def ST_NPoints(object geo_arr):
    pass

def ST_Envelope(object geo_arr):
    pass

def ST_Buffer(object geo_arr,double dfDist, n_quadrant_segments = None):
    pass

def ST_Union_Aggr(object geo_arr):
    pass

def ST_Envelope_Aggr(object geo_arr):
    pass
