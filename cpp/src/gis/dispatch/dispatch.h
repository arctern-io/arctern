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
#include <set>
#include <vector>

#include "gis/dispatch/type_scanner.h"
#include "utils/arrow_alias.h"

namespace arctern {
namespace gis {
namespace dispatch {

// deprecated
// split into [false_array, true_array]
std::array<std::shared_ptr<arrow::StringArray>, 2> WktArraySplit(
    const std::shared_ptr<arrow::Array>& geometries, const std::vector<bool>& mask);

// deprecated
// merge [false_array, true_array]
std::shared_ptr<arrow::StringArray> WktArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask);

// deprecated
// merge [false_array, true_array]
std::shared_ptr<arrow::DoubleArray> DoubleArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask);

// merge [false_array, true_array]
template <typename TypedArrowArray>
auto GenericArrayMerge(const std::array<std::shared_ptr<TypedArrowArray>, 2>& inputs,
                       const std::vector<bool>& mask) -> std::shared_ptr<TypedArrowArray>;

// return [false_array, true_array]
template <typename TypedArrowArray>
auto GenericArraySplit(const std::shared_ptr<TypedArrowArray>& geometries,
                       const std::vector<bool>& mask)
    -> std::array<std::shared_ptr<TypedArrowArray>, 2>;

template <typename RetType, typename FalseFunc, typename TrueFunc, typename Arg1>
auto UnaryMixedExecute(const std::vector<bool>& mask, FalseFunc false_func,
                       TrueFunc true_func, const std::shared_ptr<Arg1>& arg1_ptr)
    -> std::shared_ptr<RetType>;

template <typename RetType, typename FalseFunc, typename TrueFunc, typename Arg1>
auto UnaryExecute(const MaskResult& mask_result, FalseFunc false_func, TrueFunc true_func,
                  Arg1&& arg1_ptr) -> std::shared_ptr<RetType>;

template <typename RetType, typename FalseFunc, typename TrueFunc, typename Arg1,
          typename Arg2>
auto BinaryExecute(const MaskResult& mask_result, FalseFunc false_func,
                   TrueFunc true_func, Arg1&& arg1_ptr, Arg2&& arg2_ptr)
    -> std::shared_ptr<RetType>;

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern

#include "gis/dispatch/dispatch.impl.h"
