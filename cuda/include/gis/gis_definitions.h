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
namespace cuda {

class GeometryVector {
 private:
 public:
    enum class DataState : uint32_t {
        Invalid,
        Appending,
        FlatOffset_EmptyInfo,         // for calcullation: info is empty
        FlatOffset_FullInfo,          // after filling info
        PrefixSumOffset_EmptyData,    // after scan operation of meta_size/value_size
        PrefixSumOffset_FullData      // after filling the data
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

        // const version
        DEVICE_RUNNABLE const uint32_t* get_meta_ptr(int index) const {
            auto offset = meta_offsets[index];
            return metas + offset;
        }
        DEVICE_RUNNABLE const double* get_value_ptr(int index) const {
            auto offset = value_offsets[index];
            return values + offset;
        }

        // nonconst version
        DEVICE_RUNNABLE uint32_t* get_meta_ptr(int index) {
            const auto cptr = this;
            return const_cast<uint32_t*>(cptr->get_meta_ptr(index));
        }
        DEVICE_RUNNABLE double* get_value_ptr(int index) {
            const auto cptr = this;
            return const_cast<double*>(cptr->get_value_ptr(index));
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
        auto operator-> () { return ctx_.operator->(); }

     private:
        std::unique_ptr<GeoContext, Deleter> ctx_;
        explicit GeoContextHolder() : ctx_(new GeoContext) {}
        friend class GeometryVector;
    };
    GeoContextHolder CreateReadGeoContext() const;    // TODO
    GeometryVector() = default;
    GpuVector<char> EncodeToWkb() const;    // TODO

    void WkbDecodeInitalize();
    void WkbDecodeAppend(const char* bin);
    void WkbDecodeFinalize();

    void OutputInitialize(int size);
    GeoContextHolder OutputCreateGeoContext(class WorkspaceConfig& config);
    void OutputScanOn(GeoContext&);
    void OutputFinalizeFrom(const GeoContext&);

    void clear();

    int size() const { return tags_.size(); }

 private:
    GpuVector<WkbTag> tags_;
    GpuVector<uint32_t> metas_;
    GpuVector<double> values_;
    GpuVector<int> meta_offsets_;
    GpuVector<int> value_offsets_;
    DataState data_state_ = DataState::Appending;
};

//struct GeoWorkspaceConfig {
//    static constexpr int max_threads = 256 * 128;
//    int max_buffer_per_meta;     // normally 32
//    int max_buffer_per_value;    // normally 128
//    uint32_t* meta_buffer;       // size = max_threads * max_buffer_per_value
//    double* value_buffer;        // size = max_threads * max_buffer_per_value
//};
//class GeoWorkspaceConfigHolder {
// private:
//    class Deletor{
//        void operator()(GeoWorkspaceConfig*);
//    };
//    GeoWorkspaceConfigHolder(): config(new GeoWorkspaceConfig);
// public:
//    static GeoWorkspaceConfigHolder create() {
//
//    }
//
// private:
//    std::unique_ptr<GeoWorkspaceConfig, Deletor> config;
//};

using GeoContext = GeometryVector::GeoContext;

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
