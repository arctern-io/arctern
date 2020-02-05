# distutils: language = c++

from pyarrow.lib cimport *

cimport zilliz_gis_core__ as zilliz_gis_core_pxd

cdef extern from "gis.h":
    cdef shared_ptr[CArray] make_point(shared_ptr[CArray], shared_ptr[CArray])
    cdef shared_ptr[CArray] point_map(shared_ptr[CArray], shared_ptr[CArray])

cdef extern from "gis.h" namespace "zilliz::gis::cpp::gemetry":
    shared_ptr[CArray] ST_point(shared_ptr[CArray] ptr_x, shared_ptr[CArray] ptr_y)

def point_map(object arr_x,object arr_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.point_map(pyarrow_unwrap_array(arr_x),pyarrow_unwrap_array(arr_y)))
    
def make_point(object arr_x,object arr_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.make_point(pyarrow_unwrap_array(arr_x),pyarrow_unwrap_array(arr_y)))

def ST_Point(object arr_x,object arr_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Point(pyarrow_unwrap_array(arr_x),pyarrow_unwrap_array(arr_y)))

def ST_Intersection(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Intersection(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_IsValid(object geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_IsValid(pyarrow_unwrap_array(geometries)))

def ST_Equals(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Equals(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Touches(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Touches(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Overlaps(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Overlaps(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Crosses(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Crosses(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_IsSimple(object geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_IsSimple(pyarrow_unwrap_array(geometries)))

def ST_GeometryType(object geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_GeometryType(pyarrow_unwrap_array(geometries)))

def ST_MakeValid(object geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_MakeValid(pyarrow_unwrap_array(geometries)))

def ST_PrecisionReduce(object geometries,int num_dat):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_PrecisionReduce(pyarrow_unwrap_array(geometries),num_dat))

def ST_SimplifyPreserveTopology(object geometries,double distanceTolerance):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_SimplifyPreserveTopology(pyarrow_unwrap_array(geometries),distanceTolerance))
