#include "wkb/wkb_tag.h"
#include "wkb/gis_definitions.h"
#include <memory>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
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

template<typename T>
inline T*
gpu_alloc(size_t size) {
    T* ptr;
    auto err = cudaMalloc(&ptr, size * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("error with code = " + std::to_string((int)err));
    }
    return ptr;
}

template<typename T>
inline void
gpu_memcpy(T* dst, const T* src, size_t size) {
    auto err = cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        throw std::runtime_error("error with code = " + std::to_string((int)err));
    }
}


template<typename T>
inline T*
gpu_alloc_and_copy(const T* src, size_t size) {
    auto dst = gpu_alloc<T>(size);
}


GeometryVector::GPUContextHolder
GeometryVector::create_gpuctx() {
    GeometryVector::GPUContextHolder holder;
    static_assert(std::is_same<GPUVector<int>, vector<int>>::value,
                  "here use vector now");
    holder.ctx->tags = gpu_alloc() return holder;
}

void
GeometryVector::GPUContextHolder::Deleter::operator()(GPUContext* ptr) {
    if (!ptr) {
        return;
    }
}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz