set(WKB_CUDA_SOURCES
        wkb/wkb_decode.cpp
        functor/st_distance.cpp)
add_library(wkb_cuda ${WKB_CUDA_SOURCES})
target_link_libraries(wkb_cuda cudart)