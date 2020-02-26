set(gdal_tests_src
        ${unittest_srcs}
        type_scan_tests.cpp)

add_executable(gdal_tests ${gdal_tests_src})

#add_dependencies(gdal_tests googletest_ep)

target_link_libraries(gdal_tests
        windrose
        arrow
        gtest
        pthread
        gdal
        )

install(TARGETS gdal_tests DESTINATION unittest)
