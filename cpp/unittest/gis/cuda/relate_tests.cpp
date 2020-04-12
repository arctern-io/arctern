#include <gtest/gtest.h>

#include "gis/cuda/tools/relation.h"
using std::vector;
namespace cu = arctern::gis::cuda;

TEST(Relation, IsPointInLine) {
  struct Data {
    double2 point_raw;
    double2 line_beg;
    double2 line_end;
    bool std_ans;
  };
  double2 point00{0, 0};
  double2 point01{0, 1};
  double2 point11{1, 1};
  double2 point22{2, 2};
  vector<Data> datas{
      {point00, point11, point22, false},
      {point11, point00, point22, true},
      {point01, point00, point22, false},
      {point00, point00, point22, false},
  };
  for (auto data : datas) {
    auto ans = cu::IsPointInLine(data.point_raw, data.line_beg, data.line_end);
    ASSERT_EQ(ans, data.std_ans);
  }
}

