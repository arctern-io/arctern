set(WKB_CUDA_SOURCES
        wkb/wkb_decode.cpp
        wkb/wkb_transforms.cpp
        conversion/arrow_decode.cu
        conversion/arrow_decode.cpp
        conversion/arrow_encode.cu
        conversion/arrow_encode.cpp
        common/gis_definitions.cpp
        common/gis_definitions.cu
        functor/st_distance.cu
        functor/st_within.cu
        functor/st_length.cu
        functor/st_area.cu
        functor/st_envelope.cu
        functor/st_point.cu
        )
add_library(wkb_cuda ${WKB_CUDA_SOURCES})
target_link_libraries(wkb_cuda cudart gdal arrow)