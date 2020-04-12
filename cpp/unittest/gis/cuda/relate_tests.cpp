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
      {point00, point00, point22, true},
  };
  for (auto data : datas) {
    auto ans = cu::IsPointInLine(data.point_raw, data.line_beg, data.line_end);
    ASSERT_EQ(ans, data.std_ans);
  }
}

TEST(Relation, IsPointInLineString) {
  struct Data {
    double2 point;
    vector<double> lines;
    int std_count;
  };
  using vd = vector<double>;
  vector<Data> datas{
      {double2{0, 0}, {}, 0},
      {double2{0, 0}, vd{0, 0, 1, 1}, 1},
      {double2{0, 0}, vd{-1, -1, 1, 1, 2, -1}, 1},
      {double2{0, 0}, vd{-1, -1, 0, 0, 2, 0}, 2},
      {double2{0, 0}, vd{0, 0, 0, 0}, 1},
      {double2{0, 0}, vd{0, 0, 1, 1, 2, 0, 0, 0}, 2},
      {double2{2, 2}, vd{0, 0, 1, 1, 2, 0, 0, 0}, 0},
  };
  for (double x_off : {0, 1, 2}) {
    for (double y_off : {0, 10, 20}) {
      for (const auto& data : datas) {
        auto point = data.point;
        point.x += x_off;
        point.y += y_off;
        auto lines = data.lines;
        assert(lines.size() % 2 == 0);
        int size = (int)lines.size() / 2;
        for (int i = 0; i < size; ++i) {
          lines[i * 2 + 0] += x_off;
          lines[i * 2 + 1] += y_off;
        }
        auto ptr = reinterpret_cast<const double2*>(lines.data());
        auto count = cu::PointOnLineString(point, size, ptr);
        ASSERT_EQ(count, data.std_count);
      }
    }
  }
}
TEST(Relation, IsPointInLineString) {
  struct Data {
    std::vector<double> line; // sized 4
    vector<double> lnstr;
    cu::LineRelationResult std_result;
  };
  using vd = vector<double>;
  using lrr = cu::LineRelationResult;
}




