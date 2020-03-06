#pragma once
#include "gis/cuda/common/gis_definitions.h"
namespace zilliz {
namespace gis {
namespace cuda {
struct GeometryVector::GpuContext {
  WkbTag* tags = nullptr;
  uint32_t* metas = nullptr;
  double* values = nullptr;
  int* meta_offsets = nullptr;
  int* value_offsets = nullptr;
  int size = 0;
  DataState data_state = DataState::Invalid;

  DEVICE_RUNNABLE WkbTag get_tag(int index) const { return tags[index]; }

  // const pointer to start location to the index-th element
  // should be used when offsets are valid
  DEVICE_RUNNABLE const uint32_t* get_meta_ptr(int index) const {
    auto offset = meta_offsets[index];
    return metas + offset;
  }
  DEVICE_RUNNABLE const double* get_value_ptr(int index) const {
    auto offset = value_offsets[index];
    return values + offset;
  }

  // nonconst pointer to start location of the index-th element
  // should be used when offsets are valid
  DEVICE_RUNNABLE uint32_t* get_meta_ptr(int index) {
    auto offset = meta_offsets[index];
    return metas + offset;
  }
  DEVICE_RUNNABLE double* get_value_ptr(int index) {
    auto offset = value_offsets[index];
    return values + offset;
  }
  struct ConstIter {
    const uint32_t* metas;
    const double* values;
  };
  struct Iter {
    uint32_t* metas;
    double* values;
    operator ConstIter() const { return ConstIter{metas, values}; }
  };

  DEVICE_RUNNABLE Iter get_iter(int index) {
    return Iter{get_meta_ptr(index), get_value_ptr(index)};
  }

  DEVICE_RUNNABLE ConstIter get_iter(int index) const {
    return ConstIter{get_meta_ptr(index), get_value_ptr(index)};
  }
};

namespace internal {
void ExclusiveScan(int* offsets, int size);
}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
