#pragma once
#include <common/common.h>
#include <cstdint>
namespace zilliz {
namespace gis {
namespace cpp {

enum class WkbByteOrder : uint8_t { BigEndian = 0, LittleEndian = 1 };

enum class WkbCategory : uint32_t {
    Invalid = 0,
    Point = 1,
    LineString = 2,
    Polygon = 3,
    MultiPoint = 4,
    MultiLineString = 5,
    MultiPolygon = 6,
    GeometryCollection = 7,
    // TODO: TO BE CONTINUE, LAZY NOW
};

constexpr uint32_t WKBGroupBase = 1000;
enum class WkbGroup : uint32_t {
    None = 0,    // normal 2D
    Z = 1,       // XYZ
    M = 2,       // XYM
    ZM = 3       // XYZM
};

struct WkbTag {
    WkbTag() = default;
    explicit DEVICE_RUNNABLE WkbTag(uint32_t data) : data_(data) {}
    DEVICE_RUNNABLE WkbCategory get_category() {
        return static_cast<WkbCategory>(data_ % WKBGroupBase);
    }
    DEVICE_RUNNABLE WkbGroup get_group() {
        return static_cast<WkbGroup>(data_ / WKBGroupBase);
    }
    uint32_t data_;
};


}    // namespace cpp
}    // namespace gis
}    // namespace zilliz