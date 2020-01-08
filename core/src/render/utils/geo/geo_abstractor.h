#pragma once

#include <ogrsf_frmts.h>
#include <gdal_utils.h>


namespace zilliz {
namespace render {
namespace engine {

enum GeoFileType {
    kRaster,
    kVector
};

enum GeoType {
    kUnknown,
    kBuilding,
    kBlock,
    kDistrict
};

struct BoundingBox {
    double longitude_left;
    double latitude_left;
    double longitude_right;
    double latitude_right;
};

class GeoAbstractor {

 public:
    void Init() { GDALAllRegister(); }

    void set_file_path(std::string file_path) { file_path_ = file_path; }

 protected:
    GDALDataset *dataset_;
    OGRLayer *layer_;
    std::string file_path_;
    GeoFileType geo_file_type_;
    GeoType geo_type_;
};


} // namespace engine
} // namespace render
} // namespace zilliz
