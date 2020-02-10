#include "wkb/wkb_tag.h"
#include "wkb/gis_definitions.h"
#include <memory>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>

namespace zilliz {
namespace gis {
namespace cpp {

void
GeometryVector::decodeFromWKB_initialize() {
    tags_.clear();
    metas_.clear();
    values_.clear();
    meta_offsets_.clear();
    value_offsets_.clear();
    size_ = 0;
    data_state_ = DataState::Appending;
}
void
GeometryVector::decodeFromWKB_finalize(){
    meta_offsets_.push_back(0);
    value_offsets_.push_back(0);
    auto prefix_sum = [](vector<int>& vec) {
        std::exclusive_scan(vec.begin(), vec.end(), vec.begin(), 0);
    };

    data_state_ = DataState::PrefixSumOffset_FullData;
}


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
        int value_base = values_.size();
        values_.resize(values_.size() + count);
        fill<double>(2 * count, stream_iter, values_.data() + value_base);
    };

    // deal with 2D cases for now
    assert(tag.get_group() == WKB_Group::None);
    auto dimensions = 2;
    tags_.push_back(tag);
    switch (tag.get_category()) {
        case WKB_Category::Point: {
            // metas_.do_nothing()
            value_offsets_.push_back(dimensions);
            extend_values_from_stream(dimensions, 1);

            meta_offsets_.push_back(0);
            break;
        }
        case WKB_Category::LineString: {
            auto points = fetch<uint32_t>(stream_iter);
            metas_.push_back(points);
            extend_values_from_stream(dimensions, points);

            value_offsets_.push_back(1);
        }
        case WKB_Category::Polygon: {
            int total_points = 0;
            // most case 1
            auto count_sub_poly = fetch<uint32_t>(stream_iter);
            metas_.push_back(count_sub_poly);
            for (auto sub_poly = 0; sub_poly < count_sub_poly; sub_poly++) {
                auto points = fetch<uint32_t>(stream_iter);
                extend_values_from_stream(dimensions, points);
                total_points += dimensions;
                metas_.push_back(points);
            }
            meta_offsets_.push_back(1 + count_sub_poly);
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
void
gpu_free(T* ptr) {
    cudaFree(ptr);
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
    gpu_memcpy(dst, src, size);
    return dst;
}


GeometryVector::GPUContextHolder
GeometryVector::create_gpuctx() const {
    GeometryVector::GPUContextHolder holder;
    static_assert(std::is_same<GPUVector<int>, vector<int>>::value,
                  "here use vector now");
    assert(data_state_ == DataState::PrefixSumOffset_FullData);
    auto size = tags_.size();    // size_ of elements
    assert(size + 1 == meta_offsets_.size());
    assert(size + 1 == value_offsets_.size());
    assert(meta_offsets_[size] == metas_.size());
    assert(value_offsets_[size] == values_.size());

    holder.ctx->tags = gpu_alloc_and_copy(tags_.data(), tags_.size());
    holder.ctx->metas = gpu_alloc_and_copy(metas_.data(), metas_.size());
    holder.ctx->values = gpu_alloc_and_copy(values_.data(), values_.size());
    holder.ctx->meta_offsets =
        gpu_alloc_and_copy(meta_offsets_.data(), meta_offsets_.size());
    holder.ctx->value_offsets =
        gpu_alloc_and_copy(value_offsets_.data(), value_offsets_.size());
    holder.ctx->size = tags_.size();
    holder.ctx->data_state = data_state_;
    return holder;
}

void
GeometryVector::GPUContextHolder::Deleter::operator()(GPUContext* ptr) {
    if (!ptr) {
        return;
    }
    gpu_free(ptr->tags);
    gpu_free(ptr->metas);
    gpu_free(ptr->values);
    gpu_free(ptr->meta_offsets);
    gpu_free(ptr->value_offsets);
    ptr->size = 0;
    ptr->data_state = DataState::Invalid;
}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz