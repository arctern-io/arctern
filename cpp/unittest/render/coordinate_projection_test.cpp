

/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gtest/gtest.h>

#include "arrow/render_api.h"

TEST(TRANSFORM_PROJECTION_TEST, POINT_TEST) {
  // param1: wkt string    -73.984092,40.753893,-73.977588,40.756342
  std::string wkt1 = "POINT (-73.978003 40.754594)";
  std::string wkt2 = "POINT (-73.978185 40.755587)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: src_rs
  std::string src_ts = "EPSG:4326";

  // param3: dst_rs
  std::string dst_rs = "EPSG:3857";

  // param4: top_left
  std::string top_left = "POINT (-73.984092 40.756342)";

  // param5: bottom_right
  std::string bottom_right = "POINT (-73.977588 40.753893)";

  auto arr = arctern::render::transform_and_projection(string_array, src_ts, dst_rs,
                                                       bottom_right, top_left, 200, 300);

  auto str_arr = std::static_pointer_cast<arrow::StringArray>(arr);
  auto res1 = str_arr->GetString(0);
  auto res2 = str_arr->GetString(1);
  //  assert(res1 == "POINT (129 128)");
  //  assert(res2 == "POINT (180 61)");
  std::cout << res1 << std::endl;
  std::cout << res2 << std::endl;
}
