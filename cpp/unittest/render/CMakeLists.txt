#set(pointmap_test_src ${unittest_srcs} pointmap_test.cpp)
#add_executable(pointmap_tests ${pointmap_test_src})
#target_link_libraries(pointmap_tests ${unittest_libs} gdal)
#set(heatmap_test_src ${unittest_srcs} heatmap_test.cpp)
#add_executable(heatmap_tests ${heatmap_test_src})
#target_link_libraries(heatmap_tests ${unittest_libs} gdal)
#set(choroplethmap_test_src ${unittest_srcs} choropleth_map_test.cpp)
#add_executable(choroplethmap_tests ${choroplethmap_test_src})
#target_link_libraries(choroplethmap_tests ${unittest_libs} gdal)
#
#install(TARGETS pointmap_tests DESTINATION unittest)
#install(TARGETS heatmap_tests DESTINATION unittest)
#install(TARGETS choroplethmap_tests DESTINATION unittest)

set(render_tests_src
        ${unittest_srcs}
	pointmap_test.cpp
        heatmap_test.cpp
	choropleth_map_test.cpp)

add_executable(render_tests ${render_tests_src})

target_link_libraries(render_tests
        arctern
        arrow
        gtest
        pthread
        )

install(TARGETS render_tests DESTINATION unittest)

