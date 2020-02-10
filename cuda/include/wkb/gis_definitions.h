#pragma once
#include <vector>
#include <set>
#include <optional>
#include <array>
#include <tuple>
#include <memory>
#include <cassert>
using std::vector;
template<typename T>
using GPUVector = vector<T>;    // TODO: use gpu vector, now just placeholder

#include "wkb_tag.h"

namespace zilliz {
namespace gis {
namespace cpp {

class GeometryVector {
 private:
 public:
    enum class DataState : uint32_t {
        NIL,
        Appending,
        FlatOffset_EmptyData,
        PrefixSumOffset_EmptyData,
        FlatOffset_FullData,
        PrefixSumOffset_FullData
    };

    struct GPUContext {
        WKB_Tag* tags = nullptr;
        uint32_t* metas = nullptr;
        double* values = nullptr;
        int* meta_offsets = nullptr;
        int* value_offsets = nullptr;
        int size = 0;
        DataState data_state;
    };

 private:

 public:
    // just a wrapper of unique_ptr<ctx, dtor>
    class GPUContextHolder {
     public:
        const GPUContext& get() { return *ctx; }
        struct Deleter {
            void operator()(GPUContext*); // TODO
        };
     private:
        std::unique_ptr<GPUContext, Deleter> ctx;
        GPUContextHolder(): ctx(std::unique_ptr<GPUContext, Deleter>()){}
        friend class GeometryVector;
    };
    class GPUContextHolder create_gpuctx() const; // TODO
    GeometryVector() = default;
    GPUVector<char> encodeToWKB() const; // TODO
    void decodeFromWKB_append(const char* bin);

 private:
    GPUVector<WKB_Tag> tags;
    GPUVector<uint32_t> metas;
    GPUVector<double> values;
    GPUVector<int> meta_offsets;
    GPUVector<int> value_offsets;
    int size;
    DataState data_state;
};


}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
