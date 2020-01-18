set(SQLITE_CONFIGURE_ARGS
        "CPPFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 --prefix=${CMAKE_BINARY_DIR}/thirdparty")
    
set(SQLITE_SOURCE_URL
        "https://www.sqlite.org/2019/sqlite-autoconf-3300100.tar.gz")

set(SQLITE_HASH_SHA
        "8383f29d53fa1d4383e4c8eb3e087f2ed940a9e0")

ExternalProject_Add(sqlite_ep
                    URL
                    ${SQLITE_SOURCE_URL}
                    URL_HASH
                    SHA1=${SQLITE_HASH_SHA}
                    CONFIGURE_COMMAND
                    ./configure
                    CPPFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
                    --prefix=${CMAKE_BINARY_DIR}/thirdparty
                    BUILD_COMMAND
                    make
                    INSTALL_COMMAND             
                    make install
                    BUILD_IN_SOURCE             
                    1)