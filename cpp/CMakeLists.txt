project(zillizgis LANGUAGES CXX)

option(USE_GPU off)
option(BUILD_UNITTEST off)

# get build time
MACRO(GET_CURRENT_TIME CURRENT_TIME)
    execute_process(COMMAND "date" +"%Y-%m-%d %H:%M.%S" OUTPUT_VARIABLE ${CURRENT_TIME})
ENDMACRO(GET_CURRENT_TIME)

GET_CURRENT_TIME(BUILD_TIME)
string(REGEX REPLACE "\n" "" BUILD_TIME ${BUILD_TIME})
message(STATUS "Building time ${BUILD_TIME}")

# get branch name
MACRO(GET_GIT_BRANCH_NAME GIT_BRANCH_NAME)
    execute_process(COMMAND sh "-c" "git log --decorate | head -n 1 | sed 's/.*, //' | sed 's/)*//g'"
            OUTPUT_VARIABLE ${GIT_BRANCH_NAME})
ENDMACRO(GET_GIT_BRANCH_NAME)

GET_GIT_BRANCH_NAME(GIT_BRANCH_NAME)
string(REGEX REPLACE "\n" "" GIT_BRANCH_NAME ${GIT_BRANCH_NAME})
message(STATUS "Building branch ${GIT_BRANCH_NAME}")

# get last commit id
MACRO(GET_LAST_COMMIT_ID LAST_COMMIT_ID)
    execute_process(COMMAND sh "-c" "git log --decorate | head -n 1 | awk '{print $2}'"
            OUTPUT_VARIABLE ${LAST_COMMIT_ID})
ENDMACRO(GET_LAST_COMMIT_ID)

GET_LAST_COMMIT_ID(LAST_COMMIT_ID)
string(REGEX REPLACE "\n" "" LAST_COMMIT_ID ${LAST_COMMIT_ID})
message(STATUS "Building commit ${LAST_COMMIT_ID}")
if (NOT LAST_COMMIT_ID STREQUAL "")
    string(REGEX REPLACE "\n" "" LAST_COMMIT_ID ${LAST_COMMIT_ID})
    set(LAST_COMMIT_ID "${LAST_COMMIT_ID}")
else ()
    set(LAST_COMMIT_ID "Unknown")
endif ()

set(VERSION "0.1.0")

set(CMAKE_CXX_STANDARD 14)
if (USE_GPU)
    message(STATUS "Building on GPU")
    project(zillizgis LANGUAGES CUDA CXX)
    set(CMAKE_CUDA_STANDARD 14)
    add_compile_definitions(USE_GPU)
    set(CPU_OR_GPU "GPU")
else()
    set(CPU_OR_GPU "CPU")
    message(STATUS "Building on CPU")
endif ()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/common/version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/common/version.h @ONLY)

cmake_minimum_required(VERSION 3.12)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/thirdparty/include")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/thirdparty/lib")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/scripts")

include(ExternalProject)
include(ThirdParty)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message(STATUS "building GIS on x86 architecture")
    set(GIS_BUILD_ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc)")
    message(STATUS "building GIS on ppc architecture")
    set(GIS_BUILD_ARCH ppc64le)
else ()
    message(STATUS "unknown processor type")
    message(STATUS "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    set(GIS_BUILD_ARCH unknown)
endif ()

if (DEFINED APPLE)
    message(STATUS "building GIS on MacOS")
    set(GIS_BUILD_SYSTEM macos)
elseif (DEFINED UNIX)
    message(STATUS "building GIS on Unix")
    set(GIS_BUILD_SYSTEM unix)
else ()
    message(STATUS "unknown OS")
    set(GIS_BUILD_SYSTEM unknown)
endif ()

if (GIS_BUILD_ARCH STREQUAL "ppc64le")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat128")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wno-unused-variable -Wno-sign-compare -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
if (USE_GPU)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -std=c++11 -D_FORCE_INLINES -arch sm_60 --expt-extended-lambda")
endif ()

if( CMAKE_BUILD_TYPE STREQUAL "Release" )
    message(STATUS "Building Release version")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
    message(STATUS "Building Debug version")
    add_definitions(-DWRITE_IMAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
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
message(STATUS "PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")

if (USE_GPU)
    include_directories(${cuda_inc})
    link_directories(${cuda_lib})
endif ()

message(STATUS "conda_prefix = " $ENV{CONDA_PREFIX})

include_directories(${CMAKE_BINARY_DIR}/thirdparty/include)
include_directories($ENV{CONDA_PREFIX}/include)
link_directories(${CMAKE_BINARY_DIR}/thirdparty/lib)
link_directories($ENV{CONDA_PREFIX}/lib)

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

set(GIS_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/gis_env.sh.template ${CMAKE_BINARY_DIR}/scripts/gis_env.sh @ONLY)

install(DIRECTORY ${CMAKE_BINARY_DIR}/scripts/
        DESTINATION scripts
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ
        FILES_MATCHING PATTERN "*.sh")
