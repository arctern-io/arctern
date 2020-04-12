#pragma once
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
      capacity_ = std::max(new_capacity, 2 * capacity_ + 5);
      data_ = new T[capacity_];
    }
  }
  DEVICE_RUNNABLE T& operator[](int index) { return data_[index]; }
  DEVICE_RUNNABLE const T& operator[](int index) const { return data_[index]; }
  DEVICE_RUNNABLE int size() const { return size_; }
  DEVICE_RUNNABLE int capacity() const { return capacity_; }
  DEVICE_RUNNABLE void push_back(const T& x) {
    reserve(size_ + 1);
    data_[size_] = x;
    ++size_;
  }

 private:
  T* data_ = nullptr;
  int size_ = 0;
  int capacity_ = 0;
};