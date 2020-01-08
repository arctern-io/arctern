#pragma once

#include <map>

#include "render/utils/geo/geo_abstractor.h"

namespace zilliz {
namespace render {
namespace engine {


class BuildingCache {
 public:
    static BuildingCache &
    GetInstance() {
        static BuildingCache instance;
        return instance;
    }

    const std::vector<std::pair<std::string, OGRLinearRing>> &
    building_buffer() const { return building_buffer_; }

    std::vector<std::pair<std::string, OGRLinearRing>> &
    mutable_building_buffer() { return building_buffer_; }

 private:
    BuildingCache() = default;

 private:
    std::vector<std::pair<std::string, OGRLinearRing>> building_buffer_;
};


} // namespace engine
} // namespace render
} // namespace zilliz