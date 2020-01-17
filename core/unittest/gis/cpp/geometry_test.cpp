#include <gtest/gtest.h>
#include <iostream>
#include "arrow/api.h"
#include "arrow/array.h"
#include "gis/cpp/geometry/geometry.h"
using namespace zilliz::gis::cpp::gemetry;

TEST(geometry_test, make_point_from_double){
  double d_arr_x[] = {1, 2};
  double d_arr_y[] = {3, 4};

  auto res = ST_point(d_arr_x, d_arr_y, 2);

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> expect_res;
  builder.Append("POINT(1 3)");
  builder.Append("POINT(2 4)");
  builder.Finish(&expect_res);

  ASSERT_EQ(res->Equals(expect_res),true);
}

TEST(geometry_test,make_point_from_arrow){
  arrow::DoubleBuilder builder;
  builder.Append(1);
  builder.Append(2);
  std::shared_ptr<arrow::Array> ptr_x;

  builder.Finish(&ptr_x);
  builder.Append(3);
  builder.Append(4);
  std::shared_ptr<arrow::Array> ptr_y;
  builder.Finish(&ptr_y);

  auto string_ptr = ST_point(ptr_x,ptr_y);

  ASSERT_EQ(string_ptr->length(),2);

  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> expect_res;
  string_builder.Append("POINT(1 3)");
  string_builder.Append("POINT(2 4)");
  string_builder.Finish(&expect_res);

  ASSERT_EQ(string_ptr->Equals(expect_res),true);
}