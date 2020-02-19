cmake_minimum_required(VERSION 3.15)
project(zilliz_gis LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings -Xcompiler -fPIC -D_FORCE_INLINES -arch sm_60 --expt-extended-lambda")
message("CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")
set(CUDA_INCLUDE /opt/cuda/include)
set(CUDA_LIB /opt/cuda/lib64)

include_directories(${CUDA_INCLUDE})
link_directories(${CUDA_LIB})

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wconversion -pedantic")

include_directories(include)
add_subdirectory(src)
add_subdirectory(unittest)
