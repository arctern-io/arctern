from pyarrow.lib cimport *

cdef extern from "/home/zc/work/GIS/core/src/gis/make_point.cpp":
    pass

cdef extern from "/home/zc/work/GIS/core/src/gis/make_point.h":
    cdef shared_ptr[CArray] make_point(shared_ptr[CArray], shared_ptr[CArray])