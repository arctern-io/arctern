set(PROJ_ROOT   ${CMAKE_BINARY_DIR}/proj)

# note that sqlite3 must be built before this external project
set(PROJ_CMAKE_ARGS         " -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/thirdparty"
                            " -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty"
                            " -DSQLITE3_INCLUDE_DIR=${CMAKE_BINARY_DIR}/thirdparty/include"
                            " -DSQLITE3_LIBRARY=${CMAKE_BINARY_DIR}/thirdparty/lib/libsqlite3.so"
                            " -DEXE_SQLITE3=${CMAKE_BINARY_DIR}/thirdparty/bin/sqlite3"
                            " -DPROJ_TESTS=OFF")

file(MAKE_DIRECTORY "${PROJ_ROOT}/build")
        
set(PROJ_SOURCE_URL
        "https://download.osgeo.org/proj/proj-6.2.1.tar.gz")

ExternalProject_Add(proj_ep
                    URL                 ${PROJ_SOURCE_URL}
                    CONFIGURE_COMMAND   cmake ${PROJ_CMAKE_ARGS} ..
                    BINARY_DIR          ${PROJ_ROOT}/build
                    BUILD_COMMAND       make
                    INSTALL_COMMAND     make install
                    CMAKE_ARGS          ${PROJ_CMAKE_ARGS})