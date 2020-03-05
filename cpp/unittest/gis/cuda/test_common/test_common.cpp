#include "gis/cuda/test_common/test_common.h"

#include "gis/cuda/wkb/wkb_transforms.h"

namespace zilliz {
namespace gis {
namespace cuda {
// only for testing
// create Geometry from WktArray
namespace GeometryVectorFactory {

GeometryVector CreateFromWkts(const std::vector<std::string>& wkt_vec) {
  auto input = WktsToArrowWkb(wkt_vec);
  return ArrowWkbToGeometryVector(input);
}

GeometryVector CreateFromWkbs(const std::vector<std::vector<char>>& wkb_vec) {
  arrow::BinaryBuilder builder;
  for (const auto& wkb : wkb_vec) {
    auto st = builder.Append(wkb.data(), wkb.size());
    assert(st.ok());
  }
  std::shared_ptr<arrow::Array> arrow_wkb;
  auto st = builder.Finish(&arrow_wkb);
  assert(st.ok());
  auto result = ArrowWkbToGeometryVector(arrow_wkb);
  return result;
}

}  // namespace GeometryVectorFactory
}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
