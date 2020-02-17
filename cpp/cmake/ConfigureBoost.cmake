
include(FindBoostAlt)

macro(resolve_dependency DEPENDENCY_NAME)
  if(${DEPENDENCY_NAME}_SOURCE STREQUAL "AUTO")
    find_package(${DEPENDENCY_NAME} MODULE)
    if(NOT ${${DEPENDENCY_NAME}_FOUND})
      build_dependency(${DEPENDENCY_NAME})
    endif()
  elseif(${DEPENDENCY_NAME}_SOURCE STREQUAL "BUNDLED")
    build_dependency(${DEPENDENCY_NAME})
  elseif(${DEPENDENCY_NAME}_SOURCE STREQUAL "SYSTEM")
    find_package(${DEPENDENCY_NAME} REQUIRED)
  endif()
endmacro()

macro(resolve_dependency_with_version DEPENDENCY_NAME REQUIRED_VERSION)
  if(${DEPENDENCY_NAME}_SOURCE STREQUAL "AUTO")
    find_package(${DEPENDENCY_NAME} ${REQUIRED_VERSION} MODULE)
    if(NOT ${${DEPENDENCY_NAME}_FOUND})
      build_dependency(${DEPENDENCY_NAME})
    endif()
  elseif(${DEPENDENCY_NAME}_SOURCE STREQUAL "BUNDLED")
    build_dependency(${DEPENDENCY_NAME})
  elseif(${DEPENDENCY_NAME}_SOURCE STREQUAL "SYSTEM")
    find_package(${DEPENDENCY_NAME} ${REQUIRED_VERSION} REQUIRED)
  endif()
endmacro()

# ----------------------------------------------------------------------
if(DEFINED ENV{GIS_BOOST_URL})
  set(BOOST_SOURCE_URL "$ENV{GIS_BOOST_URL}")
else()
  string(REPLACE "." "_" GIS_BOOST_BUILD_VERSION_UNDERSCORES ${GIS_BOOST_BUILD_VERSION})
  set(
    BOOST_SOURCE_URL
    "https://dl.bintray.com/boostorg/release/${GIS_BOOST_BUILD_VERSION}/source/boost_${GIS_BOOST_BUILD_VERSION_UNDERSCORES}.tar.gz"
    )
endif()

macro(build_boost)

  set(BOOST_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/thirdparty")

  set(BOOST_LIB_DIR "${BOOST_PREFIX}/stage/lib")
  set(BOOST_BUILD_LINK "static")
  if("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
    set(BOOST_BUILD_VARIANT "debug")
  else()
    set(BOOST_BUILD_VARIANT "release")
  endif()
  if(MSVC)
    set(BOOST_CONFIGURE_COMMAND ".\\\\bootstrap.bat")
  else()
    set(BOOST_CONFIGURE_COMMAND "./bootstrap.sh")
  endif()
  list(APPEND BOOST_CONFIGURE_COMMAND "--prefix=${BOOST_PREFIX}"
              "--with-libraries=filesystem,regex,system")
  set(BOOST_BUILD_COMMAND "./b2" "-j8" "link=${BOOST_BUILD_LINK}"
                          "variant=${BOOST_BUILD_VARIANT}")

  set(BOOST_INSTALL_COMMAND "./b2" "install")

  if(MSVC)
    string(REGEX
           REPLACE "([0-9])$" ".\\1" BOOST_TOOLSET_MSVC_VERSION ${MSVC_TOOLSET_VERSION})
    list(APPEND BOOST_BUILD_COMMAND "toolset=msvc-${BOOST_TOOLSET_MSVC_VERSION}")
  else()
    list(APPEND BOOST_BUILD_COMMAND "cxxflags=-fPIC")
  endif()

  if(MSVC)
    string(REGEX
           REPLACE "^([0-9]+)\\.([0-9]+)\\.[0-9]+$" "\\1_\\2"
                   GIS_BOOST_BUILD_VERSION_NO_MICRO_UNDERSCORE
                   ${GIS_BOOST_BUILD_VERSION})
    set(BOOST_LIBRARY_SUFFIX "-vc${MSVC_TOOLSET_VERSION}-mt")
    if(BOOST_BUILD_VARIANT STREQUAL "debug")
      set(BOOST_LIBRARY_SUFFIX "${BOOST_LIBRARY_SUFFIX}-gd")
    endif()
    set(BOOST_LIBRARY_SUFFIX
        "${BOOST_LIBRARY_SUFFIX}-x64-${GIS_BOOST_BUILD_VERSION_NO_MICRO_UNDERSCORE}")
  else()
    set(BOOST_LIBRARY_SUFFIX "")
  endif()
  set(
    BOOST_STATIC_SYSTEM_LIBRARY
    "${BOOST_LIB_DIR}/libboost_system${BOOST_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
  set(
    BOOST_STATIC_FILESYSTEM_LIBRARY
    "${BOOST_LIB_DIR}/libboost_filesystem${BOOST_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
  set(
    BOOST_STATIC_REGEX_LIBRARY

    "${BOOST_LIB_DIR}/libboost_regex${BOOST_LIBRARY_SUFFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )
  set(BOOST_SYSTEM_LIBRARY boost_system_static)
  set(BOOST_FILESYSTEM_LIBRARY boost_filesystem_static)
  set(BOOST_REGEX_LIBRARY boost_regex_static)
  set(BOOST_BUILD_PRODUCTS ${BOOST_STATIC_SYSTEM_LIBRARY}
                           ${BOOST_STATIC_FILESYSTEM_LIBRARY}
                           ${BOOST_STATIC_REGEX_LIBRARY})
  message("BOOST INSTASLL_COMMAND, ${INSTALL_COMMAND}")
  message("BOOST INSTASLL_COMMAND, ${INSTALL_COMMAND}")
  message("BOOST INSTASLL_COMMAND, ${INSTALL_COMMAND}")
  externalproject_add(boost_ep
                      URL ${BOOST_SOURCE_URL}
                      BUILD_BYPRODUCTS ${BOOST_BUILD_PRODUCTS}
                      BUILD_IN_SOURCE 1
                      CONFIGURE_COMMAND ${BOOST_CONFIGURE_COMMAND}
                      BUILD_COMMAND ${BOOST_BUILD_COMMAND}
		      INSTALL_COMMAND ${BOOST_INSTALL_COMMAND})
  set(Boost_INCLUDE_DIR "${BOOST_PREFIX}")
  set(Boost_INCLUDE_DIRS "${BOOST_INCLUDE_DIR}")
  set(BOOST_VENDORED TRUE)
endmacro()

set(GIS_BOOST_REQUIRED_VERSION "1.67")
find_package(BoostAlt ${GIS_BOOST_REQUIRED_VERSION})

if(NOT BoostAlt_FOUND)
build_boost()
endif()

if(TARGET Boost::system)
  set(BOOST_SYSTEM_LIBRARY Boost::system)
  set(BOOST_FILESYSTEM_LIBRARY Boost::filesystem)
  set(BOOST_REGEX_LIBRARY Boost::regex)
elseif(BoostAlt_FOUND)
  set(BOOST_SYSTEM_LIBRARY ${Boost_SYSTEM_LIBRARY})
  set(BOOST_FILESYSTEM_LIBRARY ${Boost_FILESYSTEM_LIBRARY})
  set(BOOST_REGEX_LIBRARY ${Boost_REGEX_LIBRARY})
else()
  set(BOOST_SYSTEM_LIBRARY boost_system_static)
  set(BOOST_FILESYSTEM_LIBRARY boost_filesystem_static)
  set(BOOST_REGEX_LIBRARY boost_regex_static)
endif()
set(GIS_BOOST_LIBS ${BOOST_SYSTEM_LIBRARY} ${BOOST_FILESYSTEM_LIBRARY})

message(STATUS "Boost include dir: ${Boost_INCLUDE_DIR}")
message(STATUS "Boost libraries: ${GIS_BOOST_LIBS}")

include_directories(SYSTEM ${Boost_INCLUDE_DIR})
