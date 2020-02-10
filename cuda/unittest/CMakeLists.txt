set(TESTING_SRC dummy.cpp FunctorDistance_tests.cpp)
add_executable(tester ${TESTING_SRC})
target_link_libraries(tester gtest gtest_main)