cmake_minimum_required(VERSION 3.5)

project(render-engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DEPEN)

if (BUILD_UNITTEST)
    include(cmake_test_generator.cmake)
    set(GTEST_PATH "${ZDB_BUILD_HOME}/third_party/share/googletest" CACHE PATH "Path to googletest")
    add_subdirectory(${GTEST_PATH} googletest)
endif ()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message("building render_engine on x86 architecture")
    set(RENDER_ENGINE_BUILD_ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc)")
    message("building render_engine on ppc architecture")
    set(RENDER_ENGINE_BUILD_ARCH ppc64le)
else ()
    message("unknown processor type")
    message("CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}")
    set(RENDER_ENGINE_BUILD_ARCH unknown)
endif ()


if (DEFINED APPLE)
    message("building render_engine on MacOS")
    set(RENDER_ENGINE_BUILD_SYSTEM macos)
elseif (DEFINED UNIX)
    message("building render_engine on Unix")
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

#xi.ge
#set(CMAKE_BUILD_TYPE "RelWithDbgInfo")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -fPIC -Wall -Wno-unused-variable -Werror")
#MESSAGE("Building RelWithDbgInfo version")
#xi.ge

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

if (ZDB_BUILD_HOME)
else ()
    set(ZDB_BUILD_HOME "$ENV{ZDB_BUILD_HOME}")
    if ("$ENV{ZDB_BUILD_HOME}" STREQUAL "")
        message(FATAL_ERROR "set path to ZDB_BUILD_HOME")
        RETURN()
    endif ()
endif ()
message("ZDB_HOME : ${ZDB_BUILD_HOME}")

set(THIRD_PARTY_PATH ${ZDB_BUILD_HOME}/third_party)

set(THIRD_PARTY_INCLUDE ${THIRD_PARTY_PATH}/include)
set(ZLIBRARY_INCLUDE ${ZDB_BUILD_HOME}/include)
set(RENDER_ENGINEN_INCLUDE ${PROJECT_SOURCE_DIR}/include)
set(RENDER_ENGINE_SRC_INCLUDE ${PROJECT_SOURCE_DIR}/src)

set(THIRD_PARTY_LIB64 ${THIRD_PARTY_PATH}/lib64)
set(THIRD_PARTY_LIB ${THIRD_PARTY_PATH}/lib)
set(ZLIBRARY_LIB ${ZDB_BUILD_HOME}/lib)

set(cuda_inc /usr/local/cuda/include)
if (DEFINED APPLE)
    set(cuda_lib /usr/local/cuda/lib)
elseif (DEFINED UNIX)
    set(cuda_lib /usr/local/cuda/lib64)
endif ()

include_directories(${RENDER_ENGINEN_INCLUDE})
include_directories(${RENDER_ENGINE_SRC_INCLUDE})
include_directories(${PROJECT_BINARY_DIR})
message("PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")

include_directories(${ZLIBRARY_INCLUDE})
include_directories(${THIRD_PARTY_INCLUDE})
message("THIRD_PARTY_INCLUDE = ${THIRD_PARTY_INCLUDE}")

include_directories(${cuda_inc})


link_directories(${THIRD_PARTY_LIB64})
link_directories(${THIRD_PARTY_LIB})
link_directories(${ZLIBRARY_LIB})
link_directories(${cuda_lib})


add_subdirectory(..)

if (BUILD_UNITTEST)
    add_subdirectory(unittest)
endif (BUILD_UNITTEST)
unset(BUILD_UNITTEST CACHE)


#install
INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION ../..
        FILES_MATCHING
        PATTERN "*.h")


install(FILES
        package.conf
        DESTINATION package/${PROJECT_NAME})

# install images

INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/images/
        DESTINATION conf/image
        FILES_MATCHING PATTERN "*.png")


# uninstall target
if(NOT TARGET uninstall)
    configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
