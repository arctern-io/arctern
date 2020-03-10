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

