#pragma once
#include <vector>
#include <set>
#include <optional>
#include <array>
#include <tuple>
#include <cassert>
using std::vector;
template<typename T>
using GPUVector = vector<T>;    // TODO: use gpu vector, now just placeholder

#include "wkb_tag.h"

namespace zilliz {
namespace gis {
namespace cpp {
class GeometryVector {
 public:
    GeometryVector() = default;
    GPUVector<char> encodeToWKB();
    static GeometryVector decodeFromWKB(const char* bin);

 private:
    GPUVector<WKB_Tag> tags;
    GPUVector<uint32_t> metas;
    GPUVector<double> values;
    GPUVector<int> meta_offsets;
    GPUVector<int> values_offsets;
    size_t size;
};


}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
