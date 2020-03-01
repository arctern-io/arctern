#pragma once
#include "gis/cuda/common/gis_definitions.h"
namespace zilliz {
namespace gis {
namespace cuda {
struct OutputInfo {
  WkbTag tag;
  int meta_size;
  int value_size;
};

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz

#include "gis/cuda/functor/geometry_output.impl.h"
