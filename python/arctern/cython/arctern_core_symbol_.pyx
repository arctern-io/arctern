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


# render func api:
def projection(geos_list, bottom_right, top_left, height, width):
    pass

def transform_and_projection(geos_list, src_rs, dst_rs, bottom_right, top_left, height, width):
    pass

def wkt2wkb(arr_wkt):
    pass

def wkb2wkt(arr_wkb):
    pass


# render drawing api:
def point_map(vega, points_list):
    pass

def weighted_point_map(vega, points_list):
    pass

def weighted_color_point_map(vega, points_list, color_list):
    pass

def weighted_size_point_map(vega, points, size_list):
    pass

def weighted_color_size_point_map(vega, points, color_list, size_list):
    pass

def heat_map(vega, points_list, weights_list):
    pass

def choropleth_map(vega,region_boundaries_list, weights_list):
    pass

def icon_viz(vega, points_list):
    pass



# gis api
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

def ST_Within2(object geo_arr1, object geo2):
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
