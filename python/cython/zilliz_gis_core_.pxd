from pyarrow.lib cimport *

# from libcpp.string cimport *
from libcpp.string cimport *

cdef extern from "gis.h":
    cdef shared_ptr[CArray] make_point(shared_ptr[CArray], shared_ptr[CArray])
    cdef shared_ptr[CArray] point_map(shared_ptr[CArray], shared_ptr[CArray])

cdef extern from "gis.h" namespace "zilliz::gis::cpp::gemetry":
    shared_ptr[CArray] ST_point(shared_ptr[CArray] ptr_x, shared_ptr[CArray] ptr_y)