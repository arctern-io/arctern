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
#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
namespace arctern {
namespace gis {
namespace cuda {

// must free manually
template <typename T>
T* GpuAlloc(size_t size) {
  T* ptr;
  auto err = cudaMalloc(&ptr, size * sizeof(T));
  if (err != cudaSuccess) {
    throw std::runtime_error("error with code = " + std::to_string((int)err) + "<<" +
                             cudaGetErrorString(err));
  }
  return ptr;
}

template <typename T>
void GpuFree(T* ptr) {
  cudaFree(const_cast<std::remove_cv_t<T>*>(ptr));
}

template <typename T>
void GpuMemcpy(T* dst, const T* src, int size) {
  if (size == 0) {
    return;
  }
  assert(size > 0);
  auto err = cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDefault);
  if (err != cudaSuccess) {
    throw std::runtime_error("error with code = " + std::to_string((int)err) + "<<" +
                             cudaGetErrorString(err));
  }
}

// must free manually
template <typename T>
T* GpuAllocAndCopy(const T* src, int size) {
  auto dst = GpuAlloc<T>(size);
  GpuMemcpy(dst, src, size);
  return dst;
}

template <typename T>
struct GpuFreeWrapper {
  void operator()(T* ptr) { GpuFree(ptr); }
};

template <typename T>
std::unique_ptr<T, GpuFreeWrapper<T>> GpuMakeUniqueArray(int size) {
  return std::unique_ptr<T, GpuFreeWrapper<T>>(GpuAlloc<T>(size));
}

template <typename T>
auto GpuMakeUniqueArrayAndCopy(const T* src, int size)
    -> std::unique_ptr<T, GpuFreeWrapper<T>> {
  auto ptr = std::unique_ptr<T, GpuFreeWrapper<T>>(GpuAlloc<T>(size));
  GpuMemcpy(ptr.get(), src, size);
  return ptr;
}

template <typename T>
auto DebugInspect(const T* src, int size) {
  std::vector<T> tmp(size);
  GpuMemcpy(tmp.data(), src, size);
  return tmp;
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
