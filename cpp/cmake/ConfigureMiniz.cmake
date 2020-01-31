set(MINIZ_SOURCE_URL
        "https://github.com/richgel999/miniz")

set(MINIZ_CMAKE_ARGS        "-DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/thirdparty"
                            "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty"
                            "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
                            "-DCMAKE_BUILD_TYPE=Release")

ExternalProject_Add(miniz_ep
    GIT_REPOSITORY          ${MINIZ_SOURCE_URL}
    GIT_TAG                 master
    CMAKE_ARGS              ${MINIZ_CMAKE_ARGS}
)