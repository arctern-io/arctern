cmake_minimum_required(VERSION 3.14)
project(zilliz_gis LANGUAGES CUDA CXX)
find_package(CUDA)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings -Xcompiler -fPIC -D_FORCE_INLINES -arch sm_60 --expt-extended-lambda")
message("CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wconversion -pedantic")

include_directories(include)
include_directories(src)

add_subdirectory(src)
add_subdirectory(unittest)
