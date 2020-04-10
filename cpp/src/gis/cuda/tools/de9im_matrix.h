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

  DEVICE_RUNNABLE static inline State toState(char ch) { return static_cast<State>(ch); }

  DEVICE_RUNNABLE explicit constexpr Matrix(const char* text) {
    assert(text[8] == '*');
    for (int i = 0; i < 8; i++) {
      states_[i] = toState(text[i]);
    }
  }

  template <Position row, Position col>
  DEVICE_RUNNABLE void set(State state) {
    auto index = static_cast<int>(row) * 3 + static_cast<int>(col);
    if (index >= 8) {
      assert(state == State::kIgnored);
      return;
    }
    states_[index] = state;
  }

  template <Position row>
  DEVICE_RUNNABLE void set_row(const char* text) {
    assert(text[3] == '\0');
    set<row, Position::kInterier>(toState(text[0]));
    set<row, Position::kBoundry>(toState(text[1]));
    set<row, Position::kExterier>(toState(text[2]));
  }

  template <Position col>
  DEVICE_RUNNABLE void set_col(const char* text) {
    assert(text[3] == '\0');
    set<Position::kInterier, col>(toState(text[0]));
    set<Position::kBoundry, col>(toState(text[1]));
    set<Position::kExterier, col>(toState(text[2]));
  }

  DEVICE_RUNNABLE Matrix transpose() const {
    Matrix mat;
    for (int i = 0; i < 8; ++i) {
      auto ref = i / 3 + i % 3 * 3;
      mat.states_[i] = states_[ref];
    }
    return mat;
  }

 private:
  State states_[8] = {State::kInvalid};
};
constexpr Matrix INVALID_MATRIX = {};

}  // namespace de9im
