set(gis_cuda_test_srcs
        test_common/test_common.cpp
        transforms_tests.cpp
        functor_distance_tests.cpp
        functor_area_tests.cpp
        functor_point_tests.cpp
        functor_within_tests.cpp
        functor_length_tests.cpp
        functor_envelope_tests.cpp
        arrow_conversion_tests.cpp
        ${unittest_srcs}
        )
add_executable(gis_cuda_tests ${gis_cuda_test_srcs})

target_link_libraries(gis_cuda_tests arctern gtest)

install(TARGETS gis_cuda_tests DESTINATION unittest)
