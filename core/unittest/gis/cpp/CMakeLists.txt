set(geo_test_src
        ${unittest_srcs}
        geometry_test.cpp)

add_executable(geo_tests ${geo_test_src})

target_link_libraries(geo_tests
        zgeo
        arrow
        gtest
        pthread
        )