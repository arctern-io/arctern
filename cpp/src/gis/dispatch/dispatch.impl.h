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
#include <memory>
#include <utility>
#include <vector>

#include "gis/dispatch/dispatch.h"
#include "utils/arrow_alias.h"
#include "utils/check_status.h"
#include "utils/function_wrapper.h"

namespace arctern {
namespace gis {
namespace dispatch {

// return [false_array, true_array]
template <typename TypedArrowArray>
auto GenericArraySplit(const std::shared_ptr<TypedArrowArray>& geometries,
                       const std::vector<bool>& mask)
    -> std::array<std::shared_ptr<TypedArrowArray>, 2> {
  using Builder = GetArrowBuilderType<TypedArrowArray>;
  std::array<Builder, 2> builders;
  assert(mask.size() == geometries->length());
  for (auto i = 0; i < mask.size(); ++i) {
    int array_index = mask[i] ? 1 : 0;
    auto& builder = builders[array_index];
    if (geometries->IsNull(i)) {
      CHECK_ARROW(builder.AppendNull());
    } else {
      CHECK_ARROW(builder.Append(geometries->GetView(i)));
    }
  }
  std::array<std::shared_ptr<TypedArrowArray>, 2> results;
  for (auto i = 0; i < results.size(); ++i) {
    CHECK_ARROW(builders[i].Finish(&results[i]));
  }
  return results;
}

// return [false_array, true_array]
template <typename TypedArrowArray>
auto GenericArraySplitWrapper(const std::shared_ptr<arrow::Array>& geometries_raw,
                              const std::vector<bool>& mask)
    -> std::array<std::shared_ptr<TypedArrowArray>, 2> {
  auto geometries = std::static_pointer_cast<TypedArrowArray>(geometries_raw);
  return GenericArraySplit(geometries, mask);
}

// merge [false_array, true_array]
template <typename TypedArrowArray>
auto GenericArrayMerge(const std::array<std::shared_ptr<TypedArrowArray>, 2>& inputs,
                       const std::vector<bool>& mask)
    -> std::shared_ptr<TypedArrowArray> {
  assert(inputs[0]->length() + inputs[1]->length() == mask.size());
  std::array<int, 2> indexes{0, 0};
  using Builder = GetArrowBuilderType<TypedArrowArray>;
  Builder builder;
  for (auto i = 0; i < mask.size(); ++i) {
    int array_index = mask[i] ? 1 : 0;
    auto& input = inputs[array_index];
    auto index = indexes[array_index]++;
    if (input->IsNull(index)) {
      CHECK_ARROW(builder.AppendNull());
    } else {
      CHECK_ARROW(builder.Append(input->GetView(index)));
    }
  }
  std::shared_ptr<TypedArrowArray> result;
  CHECK_ARROW(builder.Finish(&result));
  return result;
}

// merge [false_array, true_array]
template <typename TypedArrowArray>
auto GenericArrayMergeWrapper(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs_raw,
    const std::vector<bool>& mask) {
  std::array<std::shared_ptr<TypedArrowArray>, 2> inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    inputs[i] = std::static_pointer_cast<TypedArrowArray>(inputs_raw[i]);
  }
  return GenericArrayMerge(inputs, mask);
}

// split into [false_array, true_array]
std::array<std::shared_ptr<arrow::StringArray>, 2> WktArraySplit(
    const std::shared_ptr<arrow::Array>& geometries, const std::vector<bool>& mask) {
  return GenericArraySplitWrapper<arrow::StringArray>(geometries, mask);
}

// merge [false_array, true_array]
std::shared_ptr<arrow::StringArray> WktArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask) {
  return GenericArrayMergeWrapper<arrow::StringArray>(inputs, mask);
}

// merge [false_array, true_array]
std::shared_ptr<arrow::DoubleArray> DoubleArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask) {
  return GenericArrayMergeWrapper<arrow::DoubleArray>(inputs, mask);
}

template <typename RetType, typename FalseFunc, typename TrueFunc, typename Arg1>
auto UnaryMixedExecute(const std::vector<bool>& mask, FalseFunc false_func,
                       TrueFunc true_func, Arg1&& arg1_ptr) -> std::shared_ptr<RetType> {
  auto split_inputs = GenericArraySplit(std::forward<Arg1>(arg1_ptr), mask);
  assert(split_inputs[1]->null_count() == 0);
  auto false_output = false_func(split_inputs[false]);
  auto true_output = true_func(split_inputs[true]);
  return GenericArrayMergeWrapper<RetType>({false_output, true_output}, mask);
}

template <typename RetType, typename FalseFunc, typename TrueFunc, typename Arg1>
auto UnaryExecute(const MaskResult& mask_result, FalseFunc false_func, TrueFunc true_func,
                  Arg1&& arg1_ptr) -> std::shared_ptr<RetType> {
  using Status = MaskResult::Status;
  switch (mask_result.get_status()) {
    case Status::kOnlyFalse: {
      return std::static_pointer_cast<RetType>(false_func(std::forward<Arg1>(arg1_ptr)));
    }
    case Status::kOnlyTrue: {
      return std::static_pointer_cast<RetType>(true_func(std::forward<Arg1>(arg1_ptr)));
    }
    case Status::kMixed: {
      return UnaryMixedExecute<RetType>(mask_result.get_mask(), false_func, true_func,
                                        std::forward<Arg1>(arg1_ptr));
    }
    default: {
      __builtin_unreachable();
      throw std::runtime_error("unreachable code");
    }
  }
}

template <typename RetType, typename FalseFunc, typename TrueFunc, typename Arg1,
          typename Arg2>
auto BinaryMixedExecute(const std::vector<bool>& mask, FalseFunc false_func,
                        TrueFunc true_func, Arg1&& arg1_ptr, Arg2&& arg2_ptr)
    -> std::shared_ptr<RetType> {
  auto split_inputs =
      std::make_tuple(GenericArraySplit(std::forward<Arg1>(arg1_ptr), mask),
                      GenericArraySplit(std::forward<Arg2>(arg2_ptr), mask));
  assert(std::get<0>(split_inputs)[true]->null_count() == 0);
  assert(std::get<1>(split_inputs)[true]->null_count() == 0);
  auto false_output =
      false_func(std::get<0>(split_inputs)[false], std::get<1>(split_inputs)[false]);
  auto true_output =
      true_func(std::get<0>(split_inputs)[true], std::get<1>(split_inputs)[true]);
  return GenericArrayMergeWrapper<RetType>({false_output, true_output}, mask);
}

template <typename RetType, typename FalseFunc, typename TrueFunc, typename Arg1,
          typename Arg2>
auto BinaryExecute(const MaskResult& mask_result, FalseFunc false_func,
                   TrueFunc true_func, Arg1&& arg1_ptr, Arg2&& arg2_ptr)
    -> std::shared_ptr<RetType> {
  using Status = MaskResult::Status;
  switch (mask_result.get_status()) {
    case Status::kOnlyFalse: {
      return std::static_pointer_cast<RetType>(
          false_func(std::forward<Arg1>(arg1_ptr), std::forward<Arg2>(arg2_ptr)));
    }
    case Status::kOnlyTrue: {
      return std::static_pointer_cast<RetType>(
          true_func(std::forward<Arg1>(arg1_ptr), std::forward<Arg2>(arg2_ptr)));
    }
    case Status::kMixed: {
      return BinaryMixedExecute<RetType>(mask_result.get_mask(), false_func, true_func,
                                         std::forward<Arg1>(arg1_ptr),
                                         std::forward<Arg2>(arg2_ptr));
    }
    default: {
      __builtin_unreachable();
      throw std::runtime_error("invalid enum");
    }
  }
}

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
