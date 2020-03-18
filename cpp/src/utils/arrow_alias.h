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
#ifdef USE_GPU
#include "gis/cuda/mock/arrow/api.h"
#else
#include <arrow/api.h>
#endif
#include <memory>
namespace arctern {

using ArrayPtr = std::shared_ptr<arrow::Array>;
using WktArrayPtr = std::shared_ptr<arrow::StringArray>;
using WkbArrayPtr = std::shared_ptr<arrow::BinaryArray>;
using DoubleArrayPtr = std::shared_ptr<arrow::DoubleArray>;
using Int32ArrayPtr = std::shared_ptr<arrow::Int32Array>;

template <typename ArrowArrayType>
using GetArrowBuilderType =
    typename arrow::TypeTraits<typename ArrowArrayType::TypeClass>::BuilderType;

}  // namespace arctern
