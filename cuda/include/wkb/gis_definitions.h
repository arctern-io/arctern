#pragma once
#include <vector>
#include <set>
#include <optional>
#include <array>
#include <tuple>
#include <cudart.h>
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
    struct GPUContext {
        WKB_Tag* tags;
        uint32_t* metas;
        double* values;
        int* meta_offsets;
        int* value_offsets;
        size_t size;
    };
    class GPUContextManager {
     public:
        ~GPUContextManager() = default;
     private:
        GPUContextManager() = default;
        GPUContextManager(const GPUContextManager&) = delete;
        GPUContextManager(GPUContextManager&&) = default;
        GPUContextManager& operator=(const GPUContextManager&) = delete;
        GPUContextManager& operator=(GPUContextManager&&) = default;
        static void destructor(GPUContext&) {
        }
        
        std::unique_ptr<GPUContext> ctx;
    };
    GPUContextManager move_to_gpu() {
        
    }
    GeometryVector() = default;
    GPUVector<char> encodeToWKB();
    void decodeFromWKB_append(const char* bin);

 private:
    GPUVector<WKB_Tag> tags;
    GPUVector<uint32_t> metas;
    GPUVector<double> values;
    GPUVector<int> meta_offsets;
    GPUVector<int> value_offsets;
    size_t size;
};


}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
