set(RAPIDJSON_SOURCE_URL
    "https://github.com/Tencent/rapidjson"
)

set(RAPIDJSON_ROOT      ${CMAKE_BINARY_DIR}/rapidjson_ep-prefix/src)
file(MAKE_DIRECTORY     ${RAPIDJSON_ROOT}/rapidjson-build)
ExternalProject_Add(rapidjson_ep
    GIT_REPOSITORY
    ${RAPIDJSON_SOURCE_URL}
    GIT_TAG
    master
    BINARY_DIR
    ${RAPIDJSON_ROOT}/rapidjson-build
    CONFIGURE_COMMAND
    cmake
    -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
    -DCMAKE_BUILD_TYPE=Release
    -DRAPIDJSON_BUILD_TESTS=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty
    ${RAPIDJSON_ROOT}/rapidjson_ep
    BUILD_COMMAND
    make
    INSTALL_COMMAND
    make install
)