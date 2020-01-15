set(pointmap_test_src ${unittest_srcs} pointmap_test.cpp)
add_executable(pointmap_tests ${pointmap_test_src})
target_link_libraries(pointmap_tests ${unittest_libs})