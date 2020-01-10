from pyarrow.lib cimport *

cdef extern from "/home/zc/work/GIS/core/src/gis/make_point.cpp":
    pass

cdef extern from "../../../core/src/gis/make_point.h":
    cdef shared_ptr[CArray] make_point(shared_ptr[CArray], shared_ptr[CArray])



cdef extern from "/home/zc/work/GIS/core/src/gis/gis_func2.cpp":
    pass

cdef extern from "../../../core/src/gis/gis_func2.h" namespace "gis":
    cdef shared_ptr[CArray] gis_func2(shared_ptr[CArray], shared_ptr[CArray])

