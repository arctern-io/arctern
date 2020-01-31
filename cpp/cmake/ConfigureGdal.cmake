set(GDAL_SOURCE_URL
        "https://github.com/OSGeo/gdal.git")
set(GDAL_ROOT       ${CMAKE_BINARY_DIR}/gdal_ep-prefix/src/)

set(GDAL_CMAKE_ARGS        "-DCMAKE_PREFIX_PATH=${CMAKE_BINARY_DIR}/thirdparty"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/thirdparty"
        "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0"
        "-DCMAKE_BUILD_TYPE=Release")

ExternalProject_Add(gdal_ep
        GIT_REPOSITORY          ${GDAL_SOURCE_URL}
        GIT_TAG                 master
        CMAKE_ARGS              ${GDAL_CMAKE_ARGS}
        BINARY_DIR
        ${GDAL_ROOT}/gdal_ep/gdal
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
        )

add_dependencies(gdal_ep proj_ep)