set(WKB_CUDA_SOURCES
        wkb/wkb_decode.cpp
        common/gis_definitions.cu
        functor/st_distance.cu
        functor/st_area.cu
        functor/st_point.cu)
add_library(wkb_cuda ${WKB_CUDA_SOURCES})
target_link_libraries(wkb_cuda cudart)