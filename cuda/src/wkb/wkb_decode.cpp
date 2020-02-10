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
        int value_base = values.size();
        values.resize(values.size() + count);
        fill<double>(2 * count, stream_iter, values.data() + value_base);
    };

    // deal with 2D cases for now
    assert(tag.get_group() == WKB_Group::None);
    auto dimensions = 2;
    tags.push_back(tag);
    switch (tag.get_category()) {
        case WKB_Category::Point: {
            // metas.do_nothing()
            value_offsets.push_back(dimensions);
            extend_values_from_stream(dimensions, 1);

            meta_offsets.push_back(0);
            break;
        }
        case WKB_Category::LineString: {
            auto points = fetch<uint32_t>(stream_iter);
            metas.push_back(points);
            extend_values_from_stream(dimensions, points);

            value_offsets.push_back(1);
        }
        case WKB_Category::Polygon: {
            int total_points = 0;
            // most case 1
            auto count_sub_poly = fetch<uint32_t>(stream_iter);
            metas.push_back(count_sub_poly);
            for (auto sub_poly = 0; sub_poly < count_sub_poly; sub_poly++) {
                auto points = fetch<uint32_t>(stream_iter);
                extend_values_from_stream(dimensions, points);
                total_points += dimensions;
                metas.push_back(points);
            }
            meta_offsets.push_back(1 + count_sub_poly);
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
void gpu_free(T* ptr) {
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
GeometryVector::create_gpuctx() const{
    GeometryVector::GPUContextHolder holder;
    static_assert(std::is_same<GPUVector<int>, vector<int>>::value,
                  "here use vector now");
    assert(data_state == DataState::PrefixSumOffset_FullData);
    auto size = tags.size(); // size of elements
    assert(size + 1 == meta_offsets.size());
    assert(size + 1 == value_offsets.size());
    assert(meta_offsets[size] == metas.size());
    assert(value_offsets[size] == values.size());

    holder.ctx->tags = gpu_alloc_and_copy(tags.data(), tags.size());
    holder.ctx->metas = gpu_alloc_and_copy(metas.data(), metas.size());
    holder.ctx->values = gpu_alloc_and_copy(values.data(), values.size());
    holder.ctx->meta_offsets = gpu_alloc_and_copy(meta_offsets.data(), meta_offsets.size());
    holder.ctx->value_offsets = gpu_alloc_and_copy(value_offsets.data(), value_offsets.size());
    holder.ctx->size = tags.size();
    holder.ctx->data_state = data_state;
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
    ptr->data_state = DataState::NIL;
}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz