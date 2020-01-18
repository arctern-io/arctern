# distutils: language = c++

from pyarrow.lib cimport *

cimport zilliz_gis_core__ as zilliz_gis_core_pxd

def make_point(object arr_x, object arr_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.make_point(pyarrow_unwrap_array(arr_x), pyarrow_unwrap_array(arr_y)))

def point_map(object arr_x, object arr_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.point_map(pyarrow_unwrap_array(arr_x), pyarrow_unwrap_array(arr_y)))

def ST_point(object arr_x,object arr_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_point(pyarrow_unwrap_array(arr_x),pyarrow_unwrap_array(arr_y)))