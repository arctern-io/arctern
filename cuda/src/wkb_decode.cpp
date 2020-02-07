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
    T tmp;
    fill<T>(1, iter, &tmp);
    return tmp;
}

GeometryVector
GeometryVector::decodeFromWKB(const char* raw_bin) {
    GeometryVector tmp;
    //
    const char* stream = raw_bin;
    auto byte_order = fetch<WKB_ByteOrder>(stream);
    assert(byte_order == WKB_ByteOrder::LittleEndian);
    auto tag = fetch<WKB_Tag>(stream);

    // deal with 2D cases for now
    assert(tag.get_group() == WKB_Group::None);

    switch (tag.get_category()) {
        case WKB_Category::Point: {
            break;
        }
        default:
            break;
    }
    return tmp;
}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz