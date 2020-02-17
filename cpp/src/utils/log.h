#pragma once

#include <memory>
#include <ostream>
#include <string>

namespace zilliz {
namespace GIS {

enum class LogLevel : int {
  DEBUG = -1,
  INFO = 0,
  WARNNING = 1,
  ERROR = 2,
  FATAL = 3
};

#define GIS_LOG_INTERNAL(level) ::zilliz::GIS::GisLog(__FILE__, __LINE__, level)
#define GIS_LOG(level) GIS_LOG_INTERNAL(::zilliz::GIS::LogLevel::##level)


} // namespace GIS
} // namespace zilliz
