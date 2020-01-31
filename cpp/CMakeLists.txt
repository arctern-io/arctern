project(zillizgis LANGUAGES CXX)

if(EXISTS ${PROJECT_SOURCE_DIR}/config.cmake)
    include(${PROJECT_SOURCE_DIR}/config.cmake)
else()
    if(EXISTS ${PROJECT_SOURCE_DIR}/config.cmake)
        include(${PROJECT_SOURCE_DIR}/config.cmake)
    endif()
endif()

set(CMAKE_CXX_STANDARD 14)
if (USE_GPU)
    project(zillizgis LANGUAGES CUDA CXX)
    set(CMAKE_CUDA_STANDARD 14)
    add_compile_definitions(USE_GPU)
endif ()

cmake_minimum_required(VERSION 3.15)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/thirdparty/include")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/thirdparty/lib")

include(ExternalProject)
include(ThirdParty)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message("building GIS on x86 architecture")
    set(GIS_BUILD_ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc)")
    message("building GIS on ppc architecture")
    set(GIS_BUILD_ARCH ppc64le)
else ()
    message("unknown processor type")
    message("CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    set(GIS_BUILD_ARCH unknown)
endif ()

if (DEFINED APPLE)
    message("building GIS on MacOS")
    set(GIS_BUILD_SYSTEM macos)
elseif (DEFINED UNIX)
    message("building GIS on Unix")
    set(GIS_BUILD_SYSTEM unix)
else ()
    message("unknown OS")
    set(GIS_BUILD_SYSTEM unknown)
endif ()

if (GIS_BUILD_ARCH STREQUAL "ppc64le")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat128")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wno-unused-variable -Wno-sign-compare -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
if (USE_GPU)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -std=c++11 -D_FORCE_INLINES -arch sm_60 --expt-extended-lambda")
endif ()

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

set(LIB_GIS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(LIB_GIS_SRC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (USE_GPU)
    set(cuda_inc /usr/local/cuda/include)
    if (DEFINED APPLE)
        set(cuda_lib /usr/local/cuda/lib)
    elseif (DEFINED UNIX)
        set(cuda_lib /usr/local/cuda/lib64)
    endif ()
endif ()


include_directories(${LIB_GIS_INCLUDE})
include_directories(${LIB_GIS_SRC_INCLUDE})
include_directories(${PROJECT_BINARY_DIR})
message("PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")

if (USE_GPU)
    include_directories(${cuda_inc})
    link_directories(${cuda_lib})
endif ()

include_directories(${CMAKE_BINARY_DIR}/thirdparty/include)
link_directories(${CMAKE_BINARY_DIR}/thirdparty/lib)

add_subdirectory(src)

if (BUILD_UNITTEST)
    add_subdirectory(unittest)
endif (BUILD_UNITTEST)
unset(BUILD_UNITTEST CACHE)

install(
    DIRECTORY ${CMAKE_BINARY_DIR}/thirdparty/lib/
    DESTINATION lib
    FILES_MATCHING PATTERN "lib*"
)
