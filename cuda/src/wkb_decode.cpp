#include "wkb/wkb_tag.h"
#include "wkb/gis_definitions.h"
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
    // decode a single polygon and append to vector
    const char* stream_iter = raw_bin;
    auto byte_order = fetch<WKB_ByteOrder>(stream_iter);
    assert(byte_order == WKB_ByteOrder::LittleEndian);
    auto tag = fetch<WKB_Tag>(stream_iter);

    auto extend_values_from_stream = [&](int dimensions, size_t points) {
        auto count = dimensions * points;
        int value_base = this->values.size();
        this->values.resize(this->values.size() + count);
        fill<double>(2 * count, stream_iter, this->values.data() + value_base);
    };

    // deal with 2D cases for now
    assert(tag.get_group() == WKB_Group::None);
    auto dimensions = 2;
    this->tags.push_back(tag);
    switch (tag.get_category()) {
        case WKB_Category::Point: {
            // this->metas.do_nothing()
            this->value_offsets.push_back(dimensions);
            extend_values_from_stream(dimensions, 1);

            this->meta_offsets.push_back(0);
            break;
        }
        case WKB_Category::LineString: {
            auto points = fetch<uint32_t>(stream_iter);
            this->metas.push_back(points);
            extend_values_from_stream(dimensions, points);

            this->value_offsets.push_back(1);
        }
        case WKB_Category::Polygon: {
            int total_points = 0;
            // most case 1
            auto count_sub_poly = fetch<uint32_t>(stream_iter);
            this->metas.push_back(count_sub_poly);
            for (auto sub_poly = 0; sub_poly < count_sub_poly; sub_poly++) {
                auto points = fetch<uint32_t>(stream_iter);
                extend_values_from_stream(dimensions, points);
                total_points += dimensions;
                this->metas.push_back(points);
            }
            this->meta_offsets.push_back(1 + count_sub_poly);
        }
        default: {
            assert(false);
            break;
        }
    }
}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz