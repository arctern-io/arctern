from pyarrow.lib cimport *

# from libcpp.string cimport *
from libcpp.string cimport *

cdef extern from "gis.h":
    shared_ptr[CArray] point_map(shared_ptr[CArray] arr_x,shared_ptr[CArray] arr_y)
    shared_ptr[CArray] make_point(shared_ptr[CArray] arr_x,shared_ptr[CArray] arr_y);

cdef extern from "gis.h" namespace "zilliz::gis":
    shared_ptr[CArray] ST_Point(const shared_ptr[CArray] &ptr_x,const shared_ptr[CArray] &ptr_y)
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