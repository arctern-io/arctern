set(gis_cuda_test_srcs dummy.cpp
        FunctorDistance_tests.cpp
        FunctorArea_tests.cpp
        FunctorPoint_tests.cpp
        ${unittest_srcs}
        )
add_executable(gis_cuda_tests ${gis_cuda_test_srcs})

target_link_libraries(gis_cuda_tests wkb_cuda gtest)

install(TARGETS gis_cuda_tests DESTINATION unittest)