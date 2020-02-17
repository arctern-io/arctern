set(geo_test_src
        ${unittest_srcs}
        geometry_test.cpp)

add_executable(geo_tests ${geo_test_src})

#add_dependencies(geo_tests googletest_ep)

target_link_libraries(geo_tests
        windrose
        arrow
        gtest
        pthread
        gdal
        )
