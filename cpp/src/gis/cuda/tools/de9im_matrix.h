#pragma once
#include <cassert>
#include <cstdint>
#include <string>

#include "gis/cuda/common/common.h"

namespace de9im {
class Matrix {
 public:
  enum class Position : uint8_t { kInterier = 0, kBoundry = 1, kExterier = 2 };
  enum class State : char {
    kInvalid = '@',

    kIgnored = '*',
    kFalse = 'F',
    kTrueGeneric = 'T',
    kDimensionZero = '0',
    kDimensionOne = '1',
    kDimensionTwo = '2',

    kComputeTrueFalse = '?',
    kComputeDimension = 'n'
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
      assert(state == State::kIgnored);
      return;
    }
    states_[index] = state;
  }

  template <Position row>
  DEVICE_RUNNABLE void set_row(const char* text) {
    assert(text[3] == '\0');
    set<row, Position::kInterier>((State)text[0]);
    set<row, Position::kBoundry>((State)text[1]);
    set<row, Position::kExterier>((State)text[2]);
  }

  template <Position col>
  DEVICE_RUNNABLE void set_col(const char* text) {
    assert(text[3] == '\0');
    set<Position::kInterier, col>((State)text[0]);
    set<Position::kBoundry, col>((State)text[1]);
    set<Position::kExterier, col>((State)text[2]);
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
      return State::kDimensionTwo;
    }
    return states_[index];
  }

 private:
  State states_[8] = {State::kInvalid};
};
constexpr Matrix INVALID_MATRIX = {};

}  // namespace de9im
