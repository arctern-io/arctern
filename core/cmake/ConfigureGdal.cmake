set(GDAL_SOURCE_URL
    "https://github.com/OSGeo/gdal/releases/download/v3.0.2/gdal-3.0.2.tar.gz")

set(GDAL_ROOT       ${CMAKE_BINARY_DIR}/gdal_ep-prefix/src/)

ExternalProject_Add(gdal_ep
    URL
    ${GDAL_SOURCE_URL}
    URL_MD5
    8a31507806b26f070858558aaad42277
    CONFIGURE_COMMAND
    ./configure
    CPPFLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
    --prefix=${CMAKE_BINARY_DIR}/thirdparty
    --with-proj=${CMAKE_BINARY_DIR}/thirdparty
    --without-libtool
    --with-zstd=no
    --with-cryptopp=no
    --with-expat=no
    --with-crypto=no
    BUILD_COMMAND
    make -j8
    INSTALL_COMMAND
    make install
    BUILD_IN_SOURCE
    1
)

add_dependencies(gdal_ep proj_ep)