PROJECT(miniz C)
cmake_minimum_required(VERSION 2.8)
if(CMAKE_BUILD_TYPE STREQUAL "")
  # CMake defaults to leaving CMAKE_BUILD_TYPE empty. This screws up
  # differentiation between debug and release builds.
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or \
CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(miniz_SOURCE miniz.c miniz_zip.c miniz_tinfl.c miniz_tdef.c)

add_library(miniz ${miniz_SOURCE})
target_include_directories(miniz PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# add_executable(miniz_tester ${MINIZ_TESTER_SRC_LIST})
# target_link_libraries(miniz_tester miniz)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
    ARCHIVE  DESTINATION lib
    LIBRARY DESTINATION lib
    )
file(GLOB INSTALL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
install(FILES ${INSTALL_HEADERS} DESTINATION include/${PROJECT_NAME})