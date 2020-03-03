set(gis_tests_src
        ${unittest_srcs}
        gis_functions_tests.cpp)

add_executable(gis_tests ${gis_tests_src})

#add_dependencies(gis_tests googletest_ep)

target_link_libraries(gis_tests
        windrose
        arrow
        gtest
        pthread
        gdal
        )

install(TARGETS gis_tests DESTINATION unittest)

add_subdirectory(gdal)
if(USE_GPU)
    add_subdirectory(cuda)
endif()
