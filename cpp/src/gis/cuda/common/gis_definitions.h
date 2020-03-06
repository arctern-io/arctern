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
#include <array>
#include <cassert>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "gis/cuda/mock/arrow/api.h"
template <typename T>
using GpuVector = std::vector<T>;  // TODO: use gpu vector, now just placeholder

#include "gis/cuda/common/function_wrapper.h"
#include "gis/cuda/wkb/wkb_tag.h"

namespace zilliz {
namespace gis {
namespace cuda {
// namespace arrow {
// class Array {
// public:
//  int length() { return 0; }
//  int null_count() { return 0; }
//};
//}  // namespace arrow

//// Not used yet, comment later
// struct GeoWorkspace {
//    static constexpr int max_threads = 256 * 128;
//    int max_buffer_per_meta = 0;         // normally 32
//    int max_buffer_per_value = 0;        // normally 128
//    uint32_t* meta_buffers = nullptr;    // size = max_threads * max_buffer_per_value
//    double* value_buffers = nullptr;     // size = max_threads * max_buffer_per_value
//    DEVICE_RUNNABLE uint32_t* get_meta_buffer(int index) {
//        assert(index < max_threads);
//        return meta_buffers + index * max_buffer_per_meta;
//    }
//    DEVICE_RUNNABLE double* get_value_buffer(int index) {
//        assert(index < max_threads);
//        return value_buffers + index * max_buffer_per_value;
//    }
//};
//
//// Not used yet, comment later
// class GeoWorkspaceHolder {
// private:
//    struct Deletor {
//        void operator()(GeoWorkspace* space) { GeoWorkspaceHolder::destruct(space); }
//    };
//    GeoWorkspaceHolder() : space_(new GeoWorkspace) {}
//    auto operator-> () { return space_.operator->(); }
//
// public:
//    static GeoWorkspaceHolder create(int max_buffer_per_meta, int max_buffer_per_value);
//    static void destruct(GeoWorkspace*);
//
// private:
//    std::unique_ptr<GeoWorkspace, Deletor> space_;
//};

// Container of the variant geometries
class GeometryVector {
 public:
  // Appending is used when decoding Wkb
  // Flat vs PrefixSum are state of meta_offsets/value_offsets
  //      FlatOffset => offsets[0, n) constains size of each element
  //      PrefixSumOffset => offsets[0, n+1) constains start location of each element
  // Info includes tags, meta_offsets, value_offsets,
  //      which is calcuated at the first pass
  // Data includes metas, values,
  //      which is calcuated at the second pass
  //      when FlatOffset, Data is always empty.
  enum class DataState : uint32_t {
    Invalid,
    Appending,
    FlatOffset_EmptyInfo,
    FlatOffset_FullInfo,
    PrefixSumOffset_EmptyData,
    PrefixSumOffset_FullData
  };

  // Geometry context,
  // raw pointers holding device memory for calculation
  // use struct to simplify data transfer in CUDA
  // fields are explained below (at class variable members declarations)
  struct GpuContext;
  struct ConstGpuContext;

 public:
  static void GpuContextDeleter(GpuContext*);
  static void GpuContextDeleter(ConstGpuContext*);
  using GpuContextHolder = UniquePtrWithDeleter<GpuContext, GpuContextDeleter>;
  using ConstGpuContextHolder = UniquePtrWithDeleter<ConstGpuContext, GpuContextDeleter>;

  ConstGpuContextHolder CreateReadGpuContext() const;  // TODO
  GeometryVector() = default;

  // STEP 1: Initialize vector with size of elements
  void OutputInitialize(int size);
  // STEP 2: Create gpu context according to the vector for cuda
  // where tags and offsets fields are uninitailized
  GpuContextHolder OutputCreateGpuContext();
  // STEP 3: Fill info(tags and offsets) to gpu_ctx using CUDA Kernels
  // where offsets[0, n) is filled with size of each element

  // STEP 4: Exclusive scan offsets[0, n+1), where offsets[n] = 0
  // then copy info(tags and scanned offsets) back to GeometryVector
  // and alloc cpu & gpu memory for next steps
  void OutputEvolveWith(GpuContext&);
  // STEP 5: Fill data(metas and values) to gpu_ctx using CUDA Kernels

  // STEP 6: Copy data(metas and values) back to GeometryVector
  void OutputFinalizeWith(GpuContext&);
  // NOTE: see functor/st_point.cu for a detailed example

  void clear();

  int size() const {
    auto tmp = tags_.size();
    assert(tmp <= std::numeric_limits<int>::max());
    return static_cast<int>(tmp);
  }

 private:
  // TODO(dog): Use Arrow Format internally
  // Currently, GpuVector contains host memory only
  // next goal should make it switchable between host and device memory.
  GpuVector<WkbTag> tags_;
  // Not including tags_, for faster access of WkbTags
  GpuVector<uint32_t> metas_;
  GpuVector<double> values_;
  // These two offsets fields contains
  //      FlatOffset => offsets[0, n) constains size of each element
  //      PrefixSumOffset => offsets[0, n+1) constains start location of each element
  GpuVector<int> meta_offsets_;
  GpuVector<int> value_offsets_;
  // This is the current state of above data containers and it companion GpuContext.
  // can only be used at assert statement for quick failure.
  // shouldn't be used to drive the state machine(e.g. switch statement)
  DataState data_state_ = DataState::Invalid;
};
using GpuContext = GeometryVector::GpuContext;
using ConstGpuContext = GeometryVector::ConstGpuContext;

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz

#include "gis/cuda/common/gis_definitions.impl.h"
