# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(ARCTERN_THIRDPARTY_DEPENDENCIES

	miniz
        stb)

message(STATUS "Using ${ARCTERN_DEPENDENCY_SOURCE} approach to find dependencies")

# For each dependency, set dependency source to global default, if unset
foreach (DEPENDENCY ${ARCTERN_THIRDPARTY_DEPENDENCIES})
    if ("${${DEPENDENCY}_SOURCE}" STREQUAL "")
        set(${DEPENDENCY}_SOURCE ${ARCTERN_DEPENDENCY_SOURCE})
    endif ()
endforeach ()

macro(build_dependency DEPENDENCY_NAME)
    if ("${DEPENDENCY_NAME}" STREQUAL "miniz")
        build_miniz()
    elseif ("${DEPENDENCY_NAME}" STREQUAL "stb")
        build_stb()
    else ()
        message(FATAL_ERROR "Unknown thirdparty dependency to build: ${DEPENDENCY_NAME}")
    endif ()
endmacro()

# ----------------------------------------------------------------------
# thirdparty directory
set(THIRDPARTY_DIR "${ARCTERN_SOURCE_DIR}/thirdparty")

macro(resolve_dependency DEPENDENCY_NAME)
    if (${DEPENDENCY_NAME}_SOURCE STREQUAL "AUTO")
        find_package(${DEPENDENCY_NAME} MODULE)
        if (NOT ${${DEPENDENCY_NAME}_FOUND})
            build_dependency(${DEPENDENCY_NAME})
        endif ()
    elseif (${DEPENDENCY_NAME}_SOURCE STREQUAL "BUNDLED")
        build_dependency(${DEPENDENCY_NAME})
    elseif (${DEPENDENCY_NAME}_SOURCE STREQUAL "SYSTEM")
        find_package(${DEPENDENCY_NAME} REQUIRED)
    endif ()
endmacro()

# ----------------------------------------------------------------------
# ExternalProject options

string(TOUPPER ${CMAKE_BUILD_TYPE} UPPERCASE_BUILD_TYPE)

set(EP_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${UPPERCASE_BUILD_TYPE}}")
set(EP_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${UPPERCASE_BUILD_TYPE}}")

# Set -fPIC on all external projects
set(EP_CXX_FLAGS "${EP_CXX_FLAGS} -fPIC")
set(EP_C_FLAGS "${EP_C_FLAGS} -fPIC")

# CC/CXX environment variables are captured on the first invocation of the
# builder (e.g make or ninja) instead of when CMake is invoked into to build
# directory. This leads to issues if the variables are exported in a subshell
# and the invocation of make/ninja is in distinct subshell without the same
# environment (CC/CXX).
set(EP_COMMON_TOOLCHAIN -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})

if (CMAKE_AR)
    set(EP_COMMON_TOOLCHAIN ${EP_COMMON_TOOLCHAIN} -DCMAKE_AR=${CMAKE_AR})
endif ()

if (CMAKE_RANLIB)
    set(EP_COMMON_TOOLCHAIN ${EP_COMMON_TOOLCHAIN} -DCMAKE_RANLIB=${CMAKE_RANLIB})
endif ()

# External projects are still able to override the following declarations.
# cmake command line will favor the last defined variable when a duplicate is
# encountered. This requires that `EP_COMMON_CMAKE_ARGS` is always the first
# argument.
set(EP_COMMON_CMAKE_ARGS
        ${EP_COMMON_TOOLCHAIN}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_FLAGS=${EP_C_FLAGS}
        -DCMAKE_C_FLAGS_${UPPERCASE_BUILD_TYPE}=${EP_C_FLAGS}
        -DCMAKE_CXX_FLAGS=${EP_CXX_FLAGS}
        -DCMAKE_CXX_FLAGS_${UPPERCASE_BUILD_TYPE}=${EP_CXX_FLAGS})

# Ensure that a default make is set
if ("${MAKE}" STREQUAL "")
    find_program(MAKE make)
endif ()

if (NOT DEFINED MAKE_BUILD_ARGS)
    set(MAKE_BUILD_ARGS "-j8")
endif ()
message(STATUS "Third Party MAKE_BUILD_ARGS = ${MAKE_BUILD_ARGS}")

# ----------------------------------------------------------------------
# Find pthreads

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ----------------------------------------------------------------------
# Versions and URLs for toolchain builds, which also can be used to configure
# offline builds

# Read toolchain versions from cpp/thirdparty/versions.txt
file(STRINGS "${THIRDPARTY_DIR}/versions.txt" TOOLCHAIN_VERSIONS_TXT)
foreach (_VERSION_ENTRY ${TOOLCHAIN_VERSIONS_TXT})
    # Exclude comments
    if (NOT _VERSION_ENTRY MATCHES "^[^#][A-Za-z0-9-_]+_VERSION=")
        continue()
    endif ()

    string(REGEX MATCH "^[^=]*" _LIB_NAME ${_VERSION_ENTRY})
    string(REPLACE "${_LIB_NAME}=" "" _LIB_VERSION ${_VERSION_ENTRY})

    # Skip blank or malformed lines
    if (${_LIB_VERSION} STREQUAL "")
        continue()
    endif ()

    # For debugging
    #message(STATUS "${_LIB_NAME}: ${_LIB_VERSION}")

    set(${_LIB_NAME} "${_LIB_VERSION}")
endforeach ()

if (DEFINED ENV{ARCTERN_MINIZ_URL})
    set(MINIZ_SOURCE_URL "$ENV{ARCTERN_MINIZ_URL}")
else ()
    set(MINIZ_SOURCE_URL
            "https://github.com/richgel999/miniz/archive/${MINIZ_VERSION}.tar.gz")
endif ()

if (DEFINED ENV{ARCTERN_STB_URL})
    set(STB_SOURCE_URL "$ENV{ARCTERN_STB_URL}")
else ()
    set(STB_SOURCE_URL
            "https://github.com/nothings/stb/archive/${STB_VERSION}.tar.gz")
endif ()

# ----------------------------------------------------------------------
# miniz

macro(build_miniz)
    message(STATUS "Building miniz-${MINIZ_VERSION} from source")
    set(MINIZ_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/miniz_ep-prefix/src/miniz_ep")
    set(MINIZ_INCLUDE_DIR "${CMAKE_BINARY_DIR}/thirdparty/include")
    set(MINIZ_SHARED_LIB
	    "${CMAKE_BINARY_DIR}/thirdparty/lib/libminiz${CMAKE_SHARED_LIBRARY_SUFFIX}")
    
    set(MINIZ_CMAKE_ARGS
#           ${EP_COMMON_CMAKE_ARGS}
             "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty"
	    "-DBUILD_SHARED_LIBS=ON"
#	    "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
            "-DCMAKE_BUILD_TYPE=Release")

    if (DEFINED ENV{ARCTERN_MINIZ_URL})
        externalproject_add(miniz_ep
                URL
                ${MINIZ_SOURCE_URL}
                BUILD_COMMAND
                ${MAKE} install
                ${MAKE_BUILD_ARGS}
                BUILD_BYPRODUCTS
                "${MINIZ_SHARED_LIB}"
                CMAKE_ARGS
                ${MINIZ_CMAKE_ARGS})
    else ()
        externalproject_add(miniz_ep
                URL
                ${MINIZ_SOURCE_URL}
		URL_HASH
		SHA1=f703804cbb325aa5ca22ebea362bb18ed73e097f
                BUILD_COMMAND
                ${MAKE} install
                ${MAKE_BUILD_ARGS}
                BUILD_BYPRODUCTS
                "${MINIZ_SHARED_LIB}"
                CMAKE_ARGS
                ${MINIZ_CMAKE_ARGS})
    endif ()

    file(MAKE_DIRECTORY "${MINIZ_INCLUDE_DIR}")
    add_library(miniz SHARED IMPORTED)
    set_target_properties(miniz
	    PROPERTIES IMPORTED_LOCATION "${MINIZ_SHARED_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${MINIZ_INCLUDE_DIR}")

    add_dependencies(miniz miniz_ep)
endmacro()

if (ARCTERN_WITH_MINIZ)
    resolve_dependency(miniz)

    get_target_property(MINIZ_INCLUDE_DIR miniz INTERFACE_INCLUDE_DIRECTORIES)
    link_directories(SYSTEM ${MINIZ_PREFIX}/lib/)
    include_directories(SYSTEM ${MINIZ_INCLUDE_DIR})
endif ()

# ----------------------------------------------------------------------
# stb

macro(build_stb)
    message(STATUS "Building stb-${STB_VERSION} from source")

    set(STB_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/stb_ep-prefix")
    set(STB_TAR_NAME "${STB_PREFIX}/stb-${STB_VERSION}.tar.gz")
    set(STB_INCLUDE_DIR "${CMAKE_BINARY_DIR}/thirdparty/include/stb")

    if (NOT EXISTS ${STB_INCLUDE_DIR})
        file(MAKE_DIRECTORY ${STB_PREFIX})
	file(DOWNLOAD ${STB_SOURCE_URL} ${STB_TAR_NAME})
	file(MAKE_DIRECTORY "${STB_INCLUDE_DIR}")
	execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${STB_TAR_NAME} WORKING_DIRECTORY ${STB_PREFIX})
	file(COPY ${STB_PREFIX}/stb-${STB_VERSION}/stb_image_write.h
	    DESTINATION ${STB_INCLUDE_DIR})
    file(COPY ${STB_PREFIX}/stb-${STB_VERSION}/stb_image.h
            DESTINATION ${STB_INCLUDE_DIR})
    endif ()
endmacro()

if (ARCTERN_WITH_STB)
    resolve_dependency(stb)
    include_directories(SYSTEM "${CMAKE_BINARY_DIR}/thirdparty/include")
endif ()
