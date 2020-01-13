# distutils: language = c++

from pyarrow.lib cimport *

cimport test_gis

def make_point2(object arr_x, object arr_y):
    return pyarrow_wrap_array(test_gis.make_point(pyarrow_unwrap_array(arr_x), pyarrow_unwrap_array(arr_y)))


def gis_func2(object arr_x, object arr_y):
    return pyarrow_wrap_array(test_gis.gis_func2(pyarrow_unwrap_array(arr_x), pyarrow_unwrap_array(arr_y)))


def point_map(object arr_x, object arr_y):
    return pyarrow_wrap_array(test_gis.point_map(pyarrow_unwrap_array(arr_x), pyarrow_unwrap_array(arr_y)))

