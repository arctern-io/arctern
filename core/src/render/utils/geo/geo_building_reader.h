#pragma once

#include "render/utils/geo/geo_abstractor.h"


namespace zilliz {
namespace render {
namespace engine {


class GeoBuildingReader : public GeoAbstractor {

 public:
    void
    Open();

    void
    Read();

    const std::string
    ReadJsonById(const int64_t &geo_id, GeoType geo_type);

//    const std::string
//    GetFilePathByGeoType(const GeoType &geo_type);

    void
    Translate(BoundingBox bound_box);

    void
    Close();

 public:
    const OGRMultiPolygon &
    buildings() const { return buildings_; }

    const std::vector<std::vector<double>> &
    polygons_xs() const { return polygons_x_; }

    const std::vector<std::vector<double>> &
    polygons_ys() const { return polygons_y_; }

 private:
    OGRMultiPolygon buildings_;

    std::vector<std::vector<double>> polygons_x_;
    std::vector<std::vector<double>> polygons_y_;
};


} // namespace engine
} // namespace render
} // namespace zilliz
