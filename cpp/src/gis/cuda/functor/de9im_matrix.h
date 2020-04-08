#pragma once
#include <cassert>
#include <cstdint>
#include <string>

#include "gis/cuda/common/common.h"

namespace de9im {
class Matrix {
 public:
  enum class Position : uint8_t { Interier = 0, Borderline = 1, Exterier = 2 };
  enum class State : uint8_t {
    Invalid = 0,

    Ignored = 0x10,
    False,
    TrueGeneric,
    DimensionZero,
    DimensionOne,
    DimensionTwo,

    ComputeIgnored = 0x20,
    ComputeTrueFalse,
    ComputeDimension
  };
  Matrix() = default;

  explicit Matrix(const std::string& text) { assert(false); }

  template <Position row, Position col>
  DEVICE_RUNNABLE State get() {
    return this->get((int)row, (int)col);
  }

  DEVICE_RUNNABLE State get(int row, int col) {
    assert(0 <= row && row < 3);
    assert(0 <= col && col < 3);
    auto index = row * 3 + col;
    if (index >= 8) {
      return State::DimensionTwo;
    }
    return states_[index];
  }

 private:
  State states_[8];
};

}  // namespace de9im
