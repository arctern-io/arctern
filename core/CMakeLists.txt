project(ZILLIZ_GIS)

cmake_minimum_required(VERSION 3.12)
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/thirdparty/include")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/thirdparty/lib")

include(ExternalProject)
include(ThirdParty)