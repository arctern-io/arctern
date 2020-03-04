#pragma once
#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/mock/arrow/api.h"
namespace zilliz {
namespace gis {
namespace cuda {

GeometryVector CreateGeometryVecorFromWkb(std::shared_ptr<arrow::Array> wkb_arrow);
std::shared_ptr<arrow::Array> ExportWkbFrom(const GeometryVector&);

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
#include "conversions.impl.h"