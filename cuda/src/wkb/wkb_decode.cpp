#include "wkb/wkb_tag.h"
#include "common/gis_definitions.h"
#include <memory>
#include <cstdlib>
#include <cstring>
//#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include "common/gpu_memory.h"

namespace zilliz {
namespace gis {
namespace cpp {

void
GeometryVector::decodeFromWKB_initialize() {
   clear();
   data_state_ = DataState::Appending;
}

void
GeometryVector::clear() {
    tags_.clear();
    metas_.clear();
    values_.clear();
    meta_offsets_.clear();
    value_offsets_.clear();
    data_state_ = DataState::Invalid;
}
void
GeometryVector::decodeFromWKB_finalize() {
    assert(data_state_ == DataState::Appending);
    assert(meta_offsets_.size() == tags_.size());
    assert(value_offsets_.size() == tags_.size());

    meta_offsets_.push_back(0);
    value_offsets_.push_back(0);

    auto prefix_sum = [](vector<int>& vec) {
        // TODO: use exclusive_scan instead
        int sum = 0;
        for (auto& x : vec) {
            auto old_sum = sum;
            sum += x;
            x = old_sum;
        }
    };

    prefix_sum(meta_offsets_);
    prefix_sum(value_offsets_);

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
    assert(data_state_ == DataState::Appending);
    const char* stream_iter = raw_bin;
    auto byte_order = fetch<WkbByteOrder>(stream_iter);
    assert(byte_order == WkbByteOrder::LittleEndian);
    auto tag = fetch<WkbTag>(stream_iter);

    auto extend_values_from_stream = [&](int dimensions, size_t points) {
        auto count = dimensions * points;
        int value_base = values_.size();
        values_.resize(values_.size() + count);
        fill<double>(count, stream_iter, values_.data() + value_base);
    };

    // deal with 2D cases for now
    assert(tag.get_group() == WkbGroup::None);
    auto dimensions = 2;
    tags_.push_back(tag);
    switch (tag.get_category()) {
        case WkbCategory::Point: {
            // metas_.do_nothing()
            extend_values_from_stream(dimensions, 1);

            meta_offsets_.push_back(0);
            value_offsets_.push_back(dimensions);
            break;
        }
        case WkbCategory::LineString: {
            auto points = fetch<uint32_t>(stream_iter);
            extend_values_from_stream(dimensions, points);

            metas_.push_back(points);
            value_offsets_.push_back(1);
            break;
        }
        case WkbCategory::Polygon: {
            int total_values = 0;
            // most case 1
            auto count_sub_poly = fetch<uint32_t>(stream_iter);
            metas_.push_back(count_sub_poly);
            for (auto sub_poly = 0; sub_poly < count_sub_poly; sub_poly++) {
                auto points = fetch<uint32_t>(stream_iter);
                extend_values_from_stream(dimensions, points);
                total_values += dimensions * points;
                metas_.push_back(points);
            }
            meta_offsets_.push_back(1 + count_sub_poly);
            value_offsets_.push_back(total_values);
            break;
        }
        default: {
            assert(false);
            break;
        }
    }
}


GeometryVector::GeoContextHolder
GeometryVector::create_gpuctx() const {
    assert(data_state_ == DataState::PrefixSumOffset_FullData);

    GeometryVector::GeoContextHolder holder;
    static_assert(std::is_same<GPUVector<int>, vector<int>>::value,
                  "here use vector now");
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
GeometryVector::GeoContextHolder::Deleter::operator()(GeoContext* ptr) {
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
