include_directories(${CMAKE_SOURCE_DIR}/src)
set(unittest_srcs
        ${CMAKE_CURRENT_SOURCE_DIR}/unittest_entry.cpp)

add_dependencies(geo_tests googletest_ep)

set(unittest_libs
        GIS
        gtest
        pthread
        )

add_subdirectory(render)
add_subdirectory(gis/cpp)
