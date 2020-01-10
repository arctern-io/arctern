set(PROJ_ROOT   ${CMAKE_BINARY_DIR}/proj)

# note that sqlite3 must be built before this external project 
 set(PROJ_CMAKE_ARGS        " -DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/thirdpart"
                            " -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty"
                            " -DSQLITE3_INCLUDE_DIR=${CMAKE_BINARY_DIR}/thirdparty/include"
                            " -DSQLITE3_LIBRARY=${CMAKE_BINARY_DIR}/thirdparty/lib/libsqlite3.so"
                            " -DEXE_SQLITE3=${CMAKE_BINARY_DIR}/thirdparty/bin/sqlite3"
                            " -DPROJ_TESTS=OFF")

set(PROJ_SOURCE_URL
        "https://download.osgeo.org/proj/proj-6.2.1.tar.gz")

ExternalProject_Add(proj_ep
                    URL                 ${PROJ_SOURCE_URL}
                    CMAKE_ARGS          ${PROJ_CMAKE_ARGS})
                
add_dependencies(proj_ep sqlite_ep)