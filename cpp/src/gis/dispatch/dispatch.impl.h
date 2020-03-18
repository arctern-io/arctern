#pragma once
#include "dispatch.h"
#include "utils/arrow_alias.h"
#include "utils/check_status.h"
#include "utils/function_wrapper.h"

namespace arctern {
namespace gis {
namespace dispatch {

// return [false_array, true_array]
inline std::array<std::shared_ptr<arrow::Array>, 2> WktArraySplit(
    const std::shared_ptr<arrow::Array>& geometries_raw, const std::vector<bool>& mask) {
  auto geometries = std::static_pointer_cast<arrow::StringArray>(geometries_raw);
  std::array<arrow::StringBuilder, 2> builders;
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
  std::array<std::shared_ptr<arrow::Array>, 2> results;
  for (auto i = 0; i < results.size(); ++i) {
    CHECK_ARROW(builders[i].Finish(&results[i]));
  }
  return results;
}

template <typename TypedArrowArray>
// merge [false_array, true_array]
std::shared_ptr<TypedArrowArray> ArrayMergeImpl(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs_raw,
    const std::vector<bool>& mask) {
  std::array<std::shared_ptr<TypedArrowArray>, 2> inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    inputs[i] = std::static_pointer_cast<TypedArrowArray>(inputs_raw[i]);
  }
  assert(inputs[0]->length() + inputs[1]->length() == mask.size());
  std::array<int, 2> indexes{0, 0};
  using TypedArrowBuilder = GetArrowBuilderType<TypedArrowArray>;
  TypedArrowBuilder builder;
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

// split into [false_array, true_array]
std::array<std::shared_ptr<arrow::Array>, 2> WktArraySplit(
    const std::shared_ptr<arrow::Array>& geometries, const std::vector<bool>& mask);

// merge [false_array, true_array]
std::shared_ptr<arrow::Array> WktArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask) {
  return ArrayMergeImpl<arrow::StringArray>(inputs, mask);
}

// merge [false_array, true_array]
std::shared_ptr<arrow::Array> DoubleArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask) {
  return ArrayMergeImpl<arrow::DoubleArray>(inputs, mask);
}
}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
