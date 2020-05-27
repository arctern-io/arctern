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

#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma once
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "gis/cuda/common/common.h"

namespace de9im {
class Matrix {
 public:
  enum class Position : uint8_t {
    kI = 0,  // Interior
    kB = 1,  // Boundary
    kE = 2,  // Exterior
  };
  enum class State : char {
    kInvalid = '\0',
    kIgnored = '*',
    kFalse = 'F',
    kTrueGeneric = 'T',
    kDim0 = '0',
    kDim1 = '1',
    kDim2 = '2',
  };

  static DEVICE_RUNNABLE bool IsTrue(State state) {
    return state != State::kInvalid && state != State::kFalse;
  }

  struct NamedStates {
    State II;
    State IB;
    State IE;
    State BI;
    State BB;
    State BE;
    State EI;
    State EB;
    constexpr State& operator[](int index) { return *(&II + index); }
    constexpr const State& operator[](int index) const { return *(&II + index); }
  } __align__(8);

 public:
  DEVICE_RUNNABLE constexpr Matrix() : states_{} {}
  constexpr Matrix(const Matrix& mat) = default;
  constexpr Matrix(Matrix&& mat) = default;
  constexpr Matrix& operator=(const Matrix& mat) = default;
  constexpr Matrix& operator=(Matrix&& mat) = default;

  DEVICE_RUNNABLE static constexpr State toState(char ch) {
    return static_cast<State>(ch);
  }

  DEVICE_RUNNABLE explicit Matrix(const char* text) : states_{} {
    assert(text[8] == '*');
    for (int i = 0; i < 8; i++) {
      states_[i] = toState(text[i]);
    }
  }

  template <Position row, Position col>
  DEVICE_RUNNABLE void set_cell(State state) {
    auto index = static_cast<int>(row) * 3 + static_cast<int>(col);
    if (index >= 8) {
      assert(state == State::kIgnored);
      return;
    }
    states_[index] = state;
  }

  DEVICE_RUNNABLE NamedStates* operator->() { return &states_; }
  DEVICE_RUNNABLE const NamedStates* operator->() const { return &states_; }

  template <Position row>
  DEVICE_RUNNABLE void set_row(const char* text) {
    assert(text[3] == '\0');
    set_cell<row, Position::kI>(toState(text[0]));
    set_cell<row, Position::kB>(toState(text[1]));
    set_cell<row, Position::kE>(toState(text[2]));
  }

  template <Position col>
  DEVICE_RUNNABLE void set_col(const char* text) {
    assert(text[3] == '\0');
    set_cell<Position::kI, col>(toState(text[0]));
    set_cell<Position::kB, col>(toState(text[1]));
    set_cell<Position::kE, col>(toState(text[2]));
  }

  DEVICE_RUNNABLE Matrix get_transpose() const {
    Matrix mat;
    for (int i = 0; i < 8; ++i) {
      auto ref = i / 3 + i % 3 * 3;
      mat.states_[i] = states_[ref];
    }
    return mat;
  }

  DEVICE_RUNNABLE friend inline bool operator==(const Matrix& a, const Matrix& b) {
    return a.get_payload() == b.get_payload();
  }

  DEVICE_RUNNABLE constexpr bool IsMatchTo(Matrix ref_matrix) const {
    for (int i = 0; i < 8; ++i) {
      auto state = states_[i];
      auto ref_state = ref_matrix.states_[i];
      assert(ref_state != State::kInvalid);
      switch (ref_state) {
        case State::kInvalid: {
          return false;
        }
        case State::kIgnored: {
          break;
        }
        case State::kTrueGeneric: {
          if (state == State::kFalse) {
            return false;
          } else {
            break;
          }
        }
        case State::kFalse:
        case State::kDim0:
        case State::kDim1:
        case State::kDim2: {
          if (state != ref_state) {
            return false;
          } else {
            break;
          }
        }
      }
    }
    return true;
  }

  DEVICE_RUNNABLE uint64_t get_payload() const {
    const void* raw_ptr = &states_;
    uint64_t payload = *reinterpret_cast<const uint64_t*>(raw_ptr);
    return payload;
  }

  friend std::ostream& operator<<(std::ostream& out, const Matrix& mat) {
    out << std::string((const char*)&mat.states_, 8);
    return out;
  }

 private:
  NamedStates states_;
};

constexpr Matrix INVALID_MATRIX;

}  // namespace de9im
