from pyarrow.lib cimport *

# from libcpp.string cimport *
from libcpp.string cimport *


#cdef extern from "make_point.cpp":
#    pass

cdef extern from "make_point.h":
    cdef shared_ptr[CArray] make_point(shared_ptr[CArray], shared_ptr[CArray])

#cdef extern from "point_map.cpp":
#    pass

cdef extern from "point_map.h":
    cdef shared_ptr[CArray] point_map(shared_ptr[CArray], shared_ptr[CArray])


