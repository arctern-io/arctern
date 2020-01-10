set(ARROW_ROOT ${CMAKE_BINARY_DIR}/arrow)

set(ARROW_CMAKE_ARGS    " -DARROW_BUILD_BENCHMARKS=OFF"
                        " -DARROW_BUILD_STATIC=OFF"
                        " -DARROW_BUILD_SHARED=ON"
                        " -DARROW_BUILD_TESTS=OFF"
                        " -DARROW_PYTHON=ON")

if(NOT CMAKE_CXX11_ABI)
    message(STATUS "ARROW: Disabling the GLIBCXX11 ABI")
    list(APPEND ARROW_CMAKE_ARGS " -DARROW_TENSORFLOW=ON")
elseif(CMAKE_CXX11_ABI)
    message(STATUS "ARROW: Enabling the GLIBCXX11 ABI")
    list(APPEND ARROW_CMAKE_ARGS " -DARROW_TENSORFLOW=OFF")
endif(NOT CMAKE_CXX11_ABI)

file(MAKE_DIRECTORY "${ARROW_ROOT}/build")
file(MAKE_DIRECTORY "${ARROW_ROOT}/install")

set(PARALLEL_BUILD -j)
if($ENV{PARALLEL_LEVEL})
    set(NUM_JOBS $ENV{PARALLEL_LEVEL})
    set(PARALLEL_BUILD "${PARALLEL_BUILD}${NUM_JOBS}")
endif($ENV{PARALLEL_LEVEL})

if(${NUM_JOBS})
    if(${NUM_JOBS} EQUAL 1)
        message(STATUS "ARROW BUILD: Enabling Sequential CMake build")
    elseif(${NUM_JOBS} GREATER 1)
        message(STATUS "ARROW BUILD: Enabling Parallel CMake build with ${NUM_JOBS} jobs")
    endif(${NUM_JOBS} EQUAL 1)
else()
    message(STATUS "ARROW BUILD: Enabling Parallel CMake build with all threads")
endif(${NUM_JOBS})

ExternalProject_Add(arrow_ep
                    GIT_REPOSITORY      https://github.com/apache/arrow.git
                    GIT_TAG             apache-arrow-0.15.1
                    SOURCE_DIR          "${ARROW_ROOT}/github"
                    SOURCE_SUBDIR       "cpp"
                    BINARY_DIR          "${ARROW_ROOT}/build"
                    INSTALL_DIR         "${ARROW_ROOT}/install"
                    CMAKE_ARGS          ${ARROW_CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty)