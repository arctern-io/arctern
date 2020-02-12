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
using GpuVector = vector<T>;    // TODO: use gpu vector, now just placeholder

#include "wkb/wkb_tag.h"

namespace zilliz {
namespace gis {
namespace cpp {

class GeometryVector {
 private:
 public:
    enum class DataState : uint32_t {
        Invalid,
        Appending,
        FlatOffset_EmptyData,
        PrefixSumOffset_EmptyData,
        FlatOffset_FullData,
        PrefixSumOffset_FullData
    };

    struct GeoContext {
        WkbTag* tags = nullptr;
        uint32_t* metas = nullptr;
        double* values = nullptr;
        int* meta_offsets = nullptr;
        int* value_offsets = nullptr;
        int size = 0;
        DataState data_state = DataState::Invalid;
        DEVICE_RUNNABLE WkbTag get_tag(int index) const { return tags[index]; }
        DEVICE_RUNNABLE const uint32_t* get_meta_ptr(int index) const {
            auto offset = meta_offsets[index];
            return metas + offset;
        }
        DEVICE_RUNNABLE const double* get_value_ptr(int index) const {
            auto offset = value_offsets[index];
            return values + offset;
        }
    };

 private:
 public:
    // just a wrapper of unique_ptr<ctx_, dtor>
    class GeoContextHolder {
     public:
        const GeoContext& get() { return *ctx_; }
        struct Deleter {
            void operator()(GeoContext*);    // TODO
        };

     private:
        std::unique_ptr<GeoContext, Deleter> ctx_;
        explicit GeoContextHolder() : ctx_(new GeoContext) {}
        friend class GeometryVector;
    };
    GeoContextHolder create_gpuctx() const;    // TODO
    GeometryVector() = default;
    GpuVector<char> EncodeToWkb() const;    // TODO

    void WkbDecodeInitalize();
    void WkbDecodeAppend(const char* bin);
    void WkbDecodeFinalize();
    void Clear();

    int size() const { return tags_.size(); }

 private:
    GpuVector<WkbTag> tags_;
    GpuVector<uint32_t> metas_;
    GpuVector<double> values_;
    GpuVector<int> meta_offsets_;
    GpuVector<int> value_offsets_;
    DataState data_state_ = DataState::Appending;
};

using GeoContext = GeometryVector::GeoContext;

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
