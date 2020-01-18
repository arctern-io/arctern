set(PROJ_SOURCE_URL "https://download.osgeo.org/proj/proj-6.2.1.tar.gz")

set(PROJ_ROOT   ${CMAKE_BINARY_DIR}/proj_ep-prefix/src)

file(MAKE_DIRECTORY     ${PROJ_ROOT}/proj-build)
ExternalProject_Add(proj_ep
    URL
    ${PROJ_SOURCE_URL}
    BINARY_DIR
    ${PROJ_ROOT}/proj-build
    CONFIGURE_COMMAND
    cmake
    -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
    -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/thirdparty
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty
    -DSQLITE3_INCLUDE_DIR=${CMAKE_BINARY_DIR}/thirdparty/include
    -DSQLITE3_LIBRARY=${CMAKE_BINARY_DIR}/thirdparty/lib/libsqlite3.so
    -DEXE_SQLITE3=${CMAKE_BINARY_DIR}/thirdparty/bin/sqlite3
    -DPROJ_TESTS=OFF
    ${PROJ_ROOT}/proj_ep
    BUILD_COMMAND
    make
    INSTALL_COMMAND
    make install
)

add_dependencies(proj_ep sqlite_ep)