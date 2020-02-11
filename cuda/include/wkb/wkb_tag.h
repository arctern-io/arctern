#pragma once
#include <common/common.h>
#include <cstdint>
namespace zilliz {
namespace gis {
namespace cpp {

enum class WKB_ByteOrder : uint8_t { BigEndian = 0, LittleEndian = 1 };

enum class WKB_Category : uint32_t {
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
enum class WKB_Group : uint32_t {
    None = 0,    // normal 2D
    Z = 1,       // XYZ
    M = 2,       // XYM
    ZM = 3       // XYZM
};

struct WKB_Tag {
    WKB_Tag() = default;
    explicit DEVICE_RUNNABLE WKB_Tag(uint32_t data) : data_(data) {}
    DEVICE_RUNNABLE WKB_Category get_category() {
        return static_cast<WKB_Category>(data_ % WKBGroupBase);
    }
    DEVICE_RUNNABLE WKB_Group get_group() {
        return static_cast<WKB_Group>(data_ / WKBGroupBase);
    }
    uint32_t data_;
};


}    // namespace cpp
}    // namespace gis
}    // namespace zilliz