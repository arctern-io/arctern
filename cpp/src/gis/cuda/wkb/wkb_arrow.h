#pragma once
#include "gis/cuda/common/gis_definitions.h"
namespace zilliz {
namespace gis {
namespace cuda {

struct ArrowContext {
  char* data;
  uint32_t* offsets;
  int size;
 public:
  DEVICE_RUNNABLE char* get_wkb_ptr(int index) { return data + index; }
  DEVICE_RUNNABLE const char* get_wkb_ptr(int index) const { return data + index; }
};

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
