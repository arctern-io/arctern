// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once
#include "gis/cuda/common/gis_definitions.h"
namespace arctern {
namespace gis {
namespace cuda {

struct GeometryVector::ConstGpuContext {
  const WkbTag* tags = nullptr;
  const uint32_t* metas = nullptr;
  const double* values = nullptr;
  const int* meta_offsets = nullptr;
  const int* value_offsets = nullptr;
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

  DEVICE_RUNNABLE int get_meta_size(int index) const {
    return meta_offsets[index + 1] - meta_offsets[index];
  }

  DEVICE_RUNNABLE int get_value_size(int index) const {
    return value_offsets[index + 1] - value_offsets[index];
  }

  struct ConstIter {
    const uint32_t* metas;
    const double* values;
    template <typename T>
    DEVICE_RUNNABLE T read_meta() {
      static_assert(sizeof(T) % sizeof(uint32_t) == 0, "mismatch");
      T tmp = *reinterpret_cast<const T*>(metas);
      metas += sizeof(T) / sizeof(uint32_t);
      return tmp;
    }
    template <typename T>
    DEVICE_RUNNABLE T read_value() {
      static_assert(sizeof(T) % sizeof(double) == 0, "mismatch");
      T tmp = *reinterpret_cast<const T*>(values);
      values += sizeof(T) / sizeof(double);
      return tmp;
    }

    template <typename T>
    DEVICE_RUNNABLE const T* read_value_ptr(int count) {
      static_assert(sizeof(T) % sizeof(double) == 0, "mismatch");
      const T* ptr = reinterpret_cast<const T*>(values);
      values += sizeof(T) / sizeof(double) * count;
      return ptr;
    }
  };

  DEVICE_RUNNABLE ConstIter get_iter(int index) const {
    return ConstIter{get_meta_ptr(index), get_value_ptr(index)};
  }
};

struct GeometryVector::GpuContext {
  WkbTag* tags = nullptr;
  uint32_t* metas = nullptr;
  double* values = nullptr;
  int* meta_offsets = nullptr;
  int* value_offsets = nullptr;
  int size = 0;
  DataState data_state = DataState::Invalid;

  DEVICE_RUNNABLE WkbTag get_tag(int index) const { return tags[index]; }

  // nonconst pointer to start location of the index-th element
  // should be used when offsets are valid
  DEVICE_RUNNABLE uint32_t* get_meta_ptr(int index) const {
    auto offset = meta_offsets[index];
    return metas + offset;
  }
  DEVICE_RUNNABLE double* get_value_ptr(int index) const {
    auto offset = value_offsets[index];
    return values + offset;
  }

  DEVICE_RUNNABLE int get_meta_size(int index) const {
    return meta_offsets[index + 1] - meta_offsets[index];
  }

  DEVICE_RUNNABLE int get_value_size(int index) const {
    return value_offsets[index + 1] - value_offsets[index];
  }

  using ConstIter = ConstGpuContext::ConstIter;
  struct Iter {
    uint32_t* metas;
    double* values;
    operator ConstIter() const { return {metas, values}; }
  };

  operator ConstGpuContext() {
    return ConstGpuContext{tags,                         //
                           metas,        values,         //
                           meta_offsets, value_offsets,  //
                           size,                         //
                           data_state};
  }

  DEVICE_RUNNABLE Iter get_iter(int index) {
    return Iter{get_meta_ptr(index), get_value_ptr(index)};
  }
};

namespace internal {
void ExclusiveScan(int* offsets, int size);
}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
