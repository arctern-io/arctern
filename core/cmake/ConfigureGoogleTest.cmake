set(GOOGLETEST_SOURCE_URL
        "https://github.com/google/googletest.git")

set(GTEST_CMAKE_ARGS        "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty"
                            "-DCMAKE_INSTALL_LIBDIR=${CMAKE_BINARY_DIR}/thirdparty/lib"
                            "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
                            "-DCMAKE_BUILD_TYPE=Release")

ExternalProject_Add(googletest_ep
    GIT_REPOSITORY              ${GOOGLETEST_SOURCE_URL}
    GIT_TAG                     master
    CMAKE_ARGS                  ${GTEST_CMAKE_ARGS}
)