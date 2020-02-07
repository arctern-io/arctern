#include "wkb_tag.h"
#include "gis_definitions.h"
#include <memory>
#include <cstdlib>
#include <cstring>


namespace zilliz {
namespace gis {
namespace cpp {

template<typename T>
static void
fill(size_t n, const char*& iter, void* dest) {
    auto len = n * sizeof(T);
    memcpy(dest, iter, len);
    iter += len;
}

template<typename T>
static T
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
    return tmp;
}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz