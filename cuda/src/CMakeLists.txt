set(WKB_CUDA_SOURCES
        gis/cuda/wkb/wkb_decode.cpp
        gis/cuda/common/gis_definitions.cu
        gis/cuda/functor/st_distance.cu
        gis/cuda/functor/st_area.cu
        gis/cuda/functor/st_point.cu)
add_library(wkb_cuda ${WKB_CUDA_SOURCES})
target_link_libraries(wkb_cuda cudart)