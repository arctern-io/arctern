# set(ARROW_SOURCE_URL
#         "https://github.com/apache/arrow.git")
#
# set(ARROW_ROOT ${CMAKE_BINARY_DIR}/arrow)
#
# set(ARROW_PREFIX ${CMAKE_BINARY_DIR}/arrow_ep-prefix/src)
# set(ARROW_BINARY_DIR ${CMAKE_BINARY_DIR}/arrow_ep-prefix/src/arrow-build)
# file(MAKE_DIRECTORY ${ARROW_BINARY_DIR})
# ExternalProject_Add(arrow_ep
#     GIT_REPOSITORY
#     ${ARROW_SOURCE_URL}
#     GIT_TAG
#     apache-arrow-0.15.1
#     BINARY_DIR
#     ${ARROW_BINARY_DIR}
#     CONFIGURE_COMMAND
#     cmake
#     -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty
#     -DARROW_BUILD_BENCHMARKS=OFF
#     -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
#     -DCMAKE_BUILD_TYPE=Release
#     -DARROW_BUILD_STATIC=OFF
#     -DARROW_BUILD_SHARED=ON
#     -DARROW_BUILD_TESTS=OFF
#     -DARROW_PYTHON=ON
#     ${ARROW_PREFIX}/arrow_ep/cpp
#     BUILD_COMMAND
#     make
#     INSTALL_COMMAND
#     make install
# )


# To keep compatibility with python bind, we need to link to arrow
# under python package
#find_package(PythonInterp)
find_package(Python3 COMPONENTS Interpreter Development)
if(NOT Python3_Interpreter_FOUND)
#if(NOT PYTHONINTERP_FOUND)
    message("cannot find python interpreter")
    exit()
endif()
message("PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")

execute_process(COMMAND "${Python3_EXECUTABLE}" "-c"
#execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import pyarrow; print(pyarrow.get_include());"
    RESULT_VARIABLE PYARROW_INCLUDE_SEARCH_SUCCESS
    OUTPUT_VARIABLE PYARROW_INCLUDE_VALUES_OUTPUT
    ERROR_VARIABLE PYARROW_INCLUDE_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(COMMAND "${Python3_EXECUTABLE}" "-c"
#execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import pyarrow; print(pyarrow.get_library_dirs());"
    RESULT_VARIABLE PYARROW_LIBRARYDIRS_SEARCH_SUCCESS
    OUTPUT_VARIABLE PYARROW_LIBRARYDIRS_VALUES_OUTPUT
    ERROR_VARIABLE PYARROW_LIBRARYDIRS_ERROR_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYARROW_INCLUDE_SEARCH_SUCCESS MATCHES 0)
    message(FATAL_ERROR "find arrow failed!")
    exit()
endif()

if(NOT PYARROW_LIBRARYDIRS_SEARCH_SUCCESS MATCHES 0)
    message(FATAL_ERROR "find arrow failed!")
    exit()
endif()

string(FIND ${PYARROW_LIBRARYDIRS_VALUES_OUTPUT} "'" ARROW_LIB_DIRS_BEGIN_QUOTE)
string(FIND ${PYARROW_LIBRARYDIRS_VALUES_OUTPUT} "'" ARROW_LIB_DIRS_END_QUOTE REVERSE)
math(EXPR ARROW_LIB_DIRS_BEGIN_POS "${ARROW_LIB_DIRS_BEGIN_QUOTE} + 1")
math(EXPR ARROW_LIB_DIRS_LENGTH "${ARROW_LIB_DIRS_END_QUOTE} - ${ARROW_LIB_DIRS_BEGIN_QUOTE} - 1")
string(SUBSTRING ${PYARROW_LIBRARYDIRS_VALUES_OUTPUT} ${ARROW_LIB_DIRS_BEGIN_POS} ${ARROW_LIB_DIRS_LENGTH} ARROW_LIB_DIRS)

include_directories(${PYARROW_INCLUDE_VALUES_OUTPUT})
link_directories(${ARROW_LIB_DIRS})
message("arrow include is here: ${PYARROW_INCLUDE_VALUES_OUTPUT}")
message("arrow library directory is here: ${ARROW_LIB_DIRS}")

# TODO: only copy files directly under lib, not deeper folder;
# the command below will mv original directory structure to destination
# install(
#     DIRECTORY ${ARROW_LIB_DIRS}/
#     DESTINATION lib
#     FILES_MATCHING PATTERN "lib*.so*"
# )
