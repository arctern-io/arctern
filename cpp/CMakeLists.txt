project(arctern LANGUAGES CXX)

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
# MACRO(GET_GIT_BRANCH_NAME GIT_BRANCH_NAME)
#     execute_process(COMMAND sh "-c" "git log --decorate | head -n 1 | sed 's/.*, //' | sed 's/)*//g'"
#             OUTPUT_VARIABLE ${GIT_BRANCH_NAME})
# ENDMACRO(GET_GIT_BRANCH_NAME)

# GET_GIT_BRANCH_NAME(GIT_BRANCH_NAME)
# string(REGEX REPLACE "\n" "" GIT_BRANCH_NAME ${GIT_BRANCH_NAME})
# message(STATUS "Building branch ${GIT_BRANCH_NAME}")

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

set(LIB_VERSION "0.1.0")

set(CMAKE_CXX_STANDARD 14)
if (USE_GPU)
    set(PYARROW_WITH_CUDA 1)
    message(STATUS "Building on GPU")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    add_compile_definitions(USE_GPU)
    set(CPU_OR_GPU "GPU")
    message("importing ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")
    message("importing ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
    include_directories(${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
    link_directories(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -D_FORCE_INLINES -arch sm_60 --expt-extended-lambda")
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

include_directories(${LIB_GIS_INCLUDE})
include_directories(${LIB_GIS_SRC_INCLUDE})
include_directories(${PROJECT_BINARY_DIR})
message(STATUS "PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")



set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
message(STATUS "conda_prefix = " ${CONDA_PREFIX})

include_directories(${CMAKE_BINARY_DIR}/thirdparty/include)
include_directories(${CONDA_PREFIX}/include)
link_directories(${CMAKE_BINARY_DIR}/thirdparty/lib)
link_directories(${CONDA_PREFIX}/lib)

add_subdirectory(src)

if (BUILD_UNITTEST)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    if (BUILD_COVERAGE STREQUAL "ON")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    endif ()
    add_subdirectory(unittest)
endif (BUILD_UNITTEST)
unset(BUILD_UNITTEST CACHE)

find_package(Python COMPONENTS Interpreter Development)
find_package(ClangTools)
set(BUILD_SUPPORT_DIR "${CMAKE_SOURCE_DIR}/build-support")

#
message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
# "make lint" target
#
if (NOT GIS_VERBOSE_LINT)
    set(GIS_LINT_QUIET "--quiet")
endif ()

if (NOT LINT_EXCLUSIONS_FILE)
    # source files matching a glob from a line in this file
    # will be excluded from linting (cpplint, clang-tidy, clang-format)
    set(LINT_EXCLUSIONS_FILE ${BUILD_SUPPORT_DIR}/lint_exclusions.txt)
endif ()

find_program(CPPLINT_BIN NAMES cpplint cpplint.py HINTS ${BUILD_SUPPORT_DIR})
message(STATUS "Found cpplint executable at ${CPPLINT_BIN}")

#
# "make lint" targets
#
add_custom_target(lint
        ${PYTHON_EXECUTABLE}
        ${BUILD_SUPPORT_DIR}/run_cpplint.py
        --cpplint_binary
        ${CPPLINT_BIN}
        --exclude_globs
        ${LINT_EXCLUSIONS_FILE}
        --source_dir
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${GIS_LINT_QUIET})

#
# "make clang-format" and "make check-clang-format" targets
#
if (${CLANG_FORMAT_FOUND})
    # runs clang format and updates files in place.
    add_custom_target(clang-format
            ${PYTHON_EXECUTABLE}
            ${BUILD_SUPPORT_DIR}/run_clang_format.py
            --clang_format_binary
            ${CLANG_FORMAT_BIN}
            --exclude_globs
            ${LINT_EXCLUSIONS_FILE}
            --source_dir
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            --fix
            ${GIS_LINT_QUIET})

    # runs clang format and exits with a non-zero exit code if any files need to be reformatted
    add_custom_target(check-clang-format
            ${PYTHON_EXECUTABLE}
            ${BUILD_SUPPORT_DIR}/run_clang_format.py
            --clang_format_binary
            ${CLANG_FORMAT_BIN}
            --exclude_globs
            ${LINT_EXCLUSIONS_FILE}
            --source_dir
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${GIS_LINT_QUIET})
endif ()

#
# "make clang-tidy" and "make check-clang-tidy" targets
#
if (${CLANG_TIDY_FOUND})
    # runs clang-tidy and attempts to fix any warning automatically
    add_custom_target(clang-tidy
            ${PYTHON_EXECUTABLE}
            ${BUILD_SUPPORT_DIR}/run_clang_tidy.py
            --clang_tidy_binary
            ${CLANG_TIDY_BIN}
            --exclude_globs
            ${LINT_EXCLUSIONS_FILE}
            --compile_commands
            ${CMAKE_BINARY_DIR}/compile_commands.json
            --source_dir
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            --fix
            ${GIS_LINT_QUIET})

    # runs clang-tidy and exits with a non-zero exit code if any errors are found.
    add_custom_target(check-clang-tidy
            ${PYTHON_EXECUTABLE}
            ${BUILD_SUPPORT_DIR}/run_clang_tidy.py
            --clang_tidy_binary
            ${CLANG_TIDY_BIN}
            --exclude_globs
            ${LINT_EXCLUSIONS_FILE}
            --compile_commands
            ${CMAKE_BINARY_DIR}/compile_commands.json
            --source_dir
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${GIS_LINT_QUIET})
endif ()

install(
    DIRECTORY ${CMAKE_BINARY_DIR}/thirdparty/lib/
    DESTINATION lib
    FILES_MATCHING PATTERN "lib*"
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/scripts/arctern_env.sh.template ${CMAKE_BINARY_DIR}/scripts/arctern_env.sh @ONLY)

install(DIRECTORY ${CMAKE_BINARY_DIR}/scripts/
        DESTINATION scripts
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ
        FILES_MATCHING PATTERN "*.sh")
