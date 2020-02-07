#include "wkb_tag.h"
#include "gis_definitions.h"
#include <memory>
#include <cstdlib>
#include <cstring>


namespace zilliz {
namespace gis {
namespace cpp {

template<typename T>
static inline void
fill(size_t count, const char*& iter, void* dest) {
    auto len = count * sizeof(T);
    memcpy(dest, iter, len);
    iter += len;
}

template<typename T>
static inline T
fetch(const char*& iter) {
    T res;
    fill<T>(1, iter, &res);
    return res;
}

void
GeometryVector::decodeFromWKB_append(const char* raw_bin) {
    //
    const char* stream_iter = raw_bin;
    auto byte_order = fetch<WKB_ByteOrder>(stream_iter);
    assert(byte_order == WKB_ByteOrder::LittleEndian);
    auto tag = fetch<WKB_Tag>(stream_iter);

    auto extend_value_from_stream = [&](size_t count) {
        auto value_base = this->values.size();
        this->values.resize(this->values.size() + count);
        fill<double>(count, stream_iter, this->values.data() + value_base);
    };

    // deal with 2D cases for now
    assert(tag.get_group() == WKB_Group::None);
    this->tags.push_back(tag);
    switch (tag.get_category()) {
        case WKB_Category::Point: {
            // this->metas nothing
            this->meta_offsets.push_back(0);

            this->value_offsets.push_back(2);
            extend_value_from_stream(2);
            break;
        }
        case WKB_Category::Polygon: {
            auto sub_polygons = fetch<uint32_t>(stream_iter);
            this->metas.push_back(sub_polygons);
            for (auto sub_poly = 0; sub_poly < sub_polygons; sub_poly++) {
                auto edges = ;
            }
        }
        default:
            break;
    }
}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz