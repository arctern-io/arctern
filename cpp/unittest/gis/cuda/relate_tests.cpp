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

TEST(Relation, LineRelateToLineString) {
  struct Data {
    std::vector<double> line;  // sized 4
    vector<double> lnstr;
    cu::LineRelationResult std_result;
  };
  thrust::complex<double> scale_factor;
  auto scale = [&scale_factor](double* ptr) {
    thrust::complex<double> raw(ptr[0], ptr[1]);
    auto tmp = scale_factor * raw;
    ptr[0] = tmp.real();
    ptr[1] = tmp.imag();
  };
  (void)scale;

  using vd = vector<double>;
  using lrr = cu::LineRelationResult;
  vector<Data> datas{
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 1, 1, 1, 0, 2, 0, 3}, lrr{1, false, -100}},
      {vd{0, 0, 0, 3}, vd{0, -100, 0, -99, 3, 3, 0, -1, 0, 1, 0, 2, 0, 4},
          lrr{1, true, -100}},
      {vd{0, 0, 0, 1}, vd{0, 1, 0, 2}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{0, 0, 2, 1, -2, 0}, lrr{0, false, 2}},
      {vd{0, 0, 0, 1}, vd{0, 0, 2, 3}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{-2, 0, 2, 0}, lrr{0, false, 1}},
      {vd{0, 0, 0, 2}, vd{0, 1, 2, 3}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{-2, 0, 2, 1}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{0, 1, 2, 2}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{0, 3, 2, 2}, lrr{-1, false, 0}},
      {vd{0, 0, 0, 1}, vd{0, 0, 0, 1}, lrr{1, true, -100}},
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 1, 0, 2, 0, 3}, lrr{1, true, -100}},
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 2, 0, 1, 0, 3}, lrr{1, true, -100}},
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 1, 1, 1, 0, 2, 0, 3, 4, 4, 0, 2, 0, 1},
       lrr{1, true, -100}},
  };
  for (auto index = 0; index < datas.size(); ++index) {
    auto data = datas[index];
    auto size = data.lnstr.size();
    assert(size % 2 == 0);
    size /= 2;
    cu::KernelBuffer buffer;
    auto result = cu::LineOnLineString((double2*)data.line.data(), size,
                                       (double2*)data.lnstr.data(), buffer);
    auto ref = data.std_result;
    ASSERT_EQ(result.II, ref.II) << index;
    ASSERT_EQ(result.is_coveredby, ref.is_coveredby) << index;
    if (ref.cross_count != -100) {
      ASSERT_EQ(result.cross_count, ref.cross_count) << index;
    }
  }
}
