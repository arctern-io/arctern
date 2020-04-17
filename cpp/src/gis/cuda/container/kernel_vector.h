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
#include <thrust/extrema.h>

#include <algorithm>

#include "gis/cuda/common/common.h"

// a mock function for std::vector
// should be used in device kernel
template <typename T>
class KernelVector {
 public:
  KernelVector() = default;
  DEVICE_RUNNABLE ~KernelVector() {
    delete data_;
    capacity_ = 0;
    size_ = 0;
  }
  DEVICE_RUNNABLE explicit KernelVector(int size) : size_(size), capacity_(size) {
    data_ = new T[capacity_];
  }
  DEVICE_RUNNABLE KernelVector(const KernelVector&) = delete;
  DEVICE_RUNNABLE KernelVector& operator=(const KernelVector&) = delete;
  // TODO: allow them
  DEVICE_RUNNABLE KernelVector(KernelVector&&) = delete;
  DEVICE_RUNNABLE KernelVector& operator=(KernelVector&&) = delete;

  DEVICE_RUNNABLE void reserve(int new_capacity) {
    if (new_capacity > capacity_) {
      delete[] data_;
      capacity_ = thrust::max(new_capacity, 2 * capacity_ + 5);
      data_ = new T[capacity_];
    }
  }
  DEVICE_RUNNABLE void clear() { size_ = 0; }
  DEVICE_RUNNABLE T& operator[](int index) { return data_[index]; }
  DEVICE_RUNNABLE const T& operator[](int index) const { return data_[index]; }
  DEVICE_RUNNABLE int size() const { return size_; }
  DEVICE_RUNNABLE int capacity() const { return capacity_; }
  DEVICE_RUNNABLE void push_back(const T& x) {
    reserve(size_ + 1);
    data_[size_] = x;
    ++size_;
  }

  // O(n^2) sort algorithm
  DEVICE_RUNNABLE void sort() {
    for (int index = 1; index < size_; index++) {
      auto value = data_[index];
      auto iter = index;
      // peek previous
      while (iter && value < data_[iter - 1]) {
        data_[iter] = data_[iter - 1];
        --iter;
      }
      data_[iter] = value;
    }
  }

 private:
  T* data_ = nullptr;
  int size_ = 0;
  int capacity_ = 0;
};
