set(ANTLR4_RUNTIME_SOURCE_URL "https://www.antlr.org/download/antlr4-cpp-runtime-4.8-source.zip")

set(ANTLR4_RUNTIME_ROOT   ${CMAKE_BINARY_DIR}/antlr4_runtime_ep-prefix/src)

ExternalProject_Add(antlr4_runtime_ep
        URL
        ${ANTLR4_RUNTIME_SOURCE_URL}
        CONFIGURE_COMMAND
        cmake
        -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/thirdparty
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty
        ${ANTLR4_RUNTIME_ROOT}/antlr4_runtime_ep
        BUILD_COMMAND
        make
        INSTALL_COMMAND
        make install
        )
