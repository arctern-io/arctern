from pyarrow.lib cimport *

# from libcpp.string cimport *
from libcpp.string cimport *

cdef extern from "gis.h" namespace "zilliz::gis":
    shared_ptr[CArray] ST_Point(const shared_ptr[CArray] &ptr_x,const shared_ptr[CArray] &ptr_y)
    shared_ptr[CArray] ST_Intersection(shared_ptr[CArray] &left_geometries,shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_IsValid(const shared_ptr[CArray] &geometries)
    shared_ptr[CArray] ST_Equals(const shared_ptr[CArray] &left_geometries, const shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_Touches(const shared_ptr[CArray] &left_geometries, const shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_Overlaps(const shared_ptr[CArray] &left_geometries,const shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_Crosses(const shared_ptr[CArray] &left_geometries, const shared_ptr[CArray] &right_geometries)
    shared_ptr[CArray] ST_IsSimple(const shared_ptr[CArray] &geometries)
#    shared_ptr[CArray] ST_PrecisionReduce(const shared_ptr[CArray] &geometries, int32_t num_dot)
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
    shared_ptr[CArray] ST_ConvexHull(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Transform(const shared_ptr[CArray] &geo_arr, const string& src_rs, const string& dst_rs)
    shared_ptr[CArray] ST_NPoints(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Envelope(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Buffer(const shared_ptr[CArray] &geo_arr, double dfDist)
    shared_ptr[CArray] ST_Buffer(const shared_ptr[CArray] &geo_arr, double dfDist, int n_quadrant_segments)
    shared_ptr[CArray] ST_Union_Aggr(const shared_ptr[CArray] &geo_arr)
    shared_ptr[CArray] ST_Envelope_Aggr(const shared_ptr[CArray] &geo_arr)
