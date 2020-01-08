cmake_minimum_required(VERSION 3.5)

project(libGIS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DEPEN)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message("building libGIS on x86 architecture")
    set(RENDER_ENGINE_BUILD_ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc)")
    message("building libGIS on ppc architecture")
    set(RENDER_ENGINE_BUILD_ARCH ppc64le)
else ()
    message("unknown processor type")
    message("CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    set(RENDER_ENGINE_BUILD_ARCH unknown)
endif ()


if (DEFINED APPLE)
    message("building libGIS on MacOS")
    set(RENDER_ENGINE_BUILD_SYSTEM macos)
elseif (DEFINED UNIX)
    message("building libGIS on Unix")
    set(RENDER_ENGINE_BUILD_SYSTEM unix)
else ()
    message("unknown OS")
    set(RENDER_ENGINE_BUILD_SYSTEM unknown)
endif ()

if (RENDER_ENGINE_BUILD_ARCH STREQUAL "ppc64le")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat128")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wno-unused-variable -Wno-sign-compare -Werror")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -std=c++11 -D_FORCE_INLINES -arch sm_60 --expt-extended-lambda")

if( CMAKE_BUILD_TYPE STREQUAL "Release" )
    message("Building Release version")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
    message("Building Debug version")
    add_definitions(-DWRITE_IMAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

message("CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
# Arch info over

set(LIB_GIS_INCLUDE ${PROJECT_SOURCE_DIR}/src)
set(LIB_GIS_SRC_INCLUDE ${PROJECT_SOURCE_DIR}/src)

set(cuda_inc /usr/local/cuda/include)
if (DEFINED APPLE)
    set(cuda_lib /usr/local/cuda/lib)
elseif (DEFINED UNIX)
    set(cuda_lib /usr/local/cuda/lib64)
endif ()

include_directories(${LIB_GIS_INCLUDE})
include_directories(${LIB_GIS_SRC_INCLUDE})
include_directories(${PROJECT_BINARY_DIR})
message("PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")

include_directories(${cuda_inc})
link_directories(${cuda_lib})


add_subdirectory(src)

#install
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/src DESTINATION .
        FILES_MATCHING
        PATTERN "*.h")


install(FILES
        package.conf
        DESTINATION package/${PROJECT_NAME})

# install images

INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/images/
        DESTINATION conf/image
        FILES_MATCHING PATTERN "*.png")

