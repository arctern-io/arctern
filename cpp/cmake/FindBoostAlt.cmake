if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  # In older versions of CMake (such as 3.2), the system paths for Boost will
  # be looked in first even if we set $BOOST_ROOT or pass -DBOOST_ROOT
  set(Boost_NO_SYSTEM_PATHS ON)
endif()

set(BoostAlt_FIND_VERSION_OPTIONS)
if(BoostAlt_FIND_VERSION)
  list(APPEND BoostAlt_FIND_VERSION_OPTIONS ${BoostAlt_FIND_VERSION})
endif()
if(BoostAlt_FIND_REQUIRED)
  list(APPEND BoostAlt_FIND_VERSION_OPTIONS REQUIRED)
endif()
if(BoostAlt_FIND_QUIETLY)
  list(APPEND BoostAlt_FIND_VERSION_OPTIONS QUIET)
endif()

set(Boost_USE_STATIC_LIBS ON)
set(BUILD_SHARED_LIBS OFF)
find_package(Boost ${BoostAlt_FIND_VERSION_OPTIONS}
             COMPONENTS regex system filesystem)

if(Boost_FOUND)
  set(BoostAlt_FOUND ON)
else()
  set(BoostAlt_FOUND OFF)
endif()
