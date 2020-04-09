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
    Invalid = '@',

    Ignored = '*',
    False = 'F',
    TrueGeneric = 'T',
    DimensionZero = '0',
    DimensionOne = '1',
    DimensionTwo = '2',

    ComputeTrueFalse = '?',
    ComputeDimension = 'n'
  };
  Matrix() = default;

  DEVICE_RUNNABLE constexpr Matrix(const char* text) {
    assert(text[8] == '*');
    for (int i = 0; i < 8; i++) {
      states_[i] = (State)text[i];
    }
  }

  template <Position row, Position col>
  DEVICE_RUNNABLE State get() {
    return this->get((int)row, (int)col);
  }

  DEVICE_RUNNABLE void set(int row, int col, State state) {
    assert(0 <= row && row < 3);
    assert(0 <= col && col < 3);
    auto index = row * 3 + col;
    if (index >= 8) {
      return;
    }
    states_[index] = state;
  }

  template <Position row, Position col>
  DEVICE_RUNNABLE void set(State state) {
    set((int)row, (int)col, state);
  };

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
  State states_[8] = {State::Invalid};
};

constexpr Matrix INVALID_MATRIX = {};

}  // namespace de9im
