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
#include "common/function_wrapper.h"

namespace zilliz {
namespace gis {
namespace cuda {

// Not used yet, comment later
struct GeoWorkspace {
    static constexpr int max_threads = 256 * 128;
    int max_buffer_per_meta = 0;         // normally 32
    int max_buffer_per_value = 0;        // normally 128
    uint32_t* meta_buffers = nullptr;    // size = max_threads * max_buffer_per_value
    double* value_buffers = nullptr;     // size = max_threads * max_buffer_per_value
    DEVICE_RUNNABLE uint32_t* get_meta_buffer(int index) {
        assert(index < max_threads);
        return meta_buffers + index;
    }
    DEVICE_RUNNABLE double* get_value_buffer(int index) {
        assert(index < max_threads);
        return value_buffers + index;
    }
};

//// Not used yet, comment later
//class GeoWorkspaceHolder {
// private:
//    struct Deletor {
//        void operator()(GeoWorkspace* space) { GeoWorkspaceHolder::destruct(space); }
//    };
//    GeoWorkspaceHolder() : space_(new GeoWorkspace) {}
//    auto operator-> () { return space_.operator->(); }
//
// public:
//    static GeoWorkspaceHolder create(int max_buffer_per_meta, int max_buffer_per_value);
//    static void destruct(GeoWorkspace*);
//
// private:
//    std::unique_ptr<GeoWorkspace, Deletor> space_;
//};

// Container of the variant geometries
class GeometryVector {
 private:
 public:
    // Appending is used when decoding Wkb
    // Flat vs PrefixSum are explained below (at data_state_ declaration)
    // Info includes tags, meta_offsets, value_offsets,
    //      which is calcuated at the first pass
    // Data includes metas, values,
    //      which is calcuated at the second pass
    //      when FlatOffset, Data is always empty.
    enum class DataState : uint32_t {
        Invalid,
        Appending,
        FlatOffset_EmptyInfo,         // for calcullation: info is empty
        FlatOffset_FullInfo,          // after filling info
        PrefixSumOffset_EmptyData,    // after scan operation of meta_size/value_size
        PrefixSumOffset_FullData      // after filling the data
    };

    // Geometry context,
    // raw pointers holding device memory for calculation
    // use struct to simplify data transfer in CUDA
    // fields are explained below (at class variable members declarations)
    struct GpuContext {
        WkbTag* tags = nullptr;
        uint32_t* metas = nullptr;
        double* values = nullptr;
        int* meta_offsets = nullptr;
        int* value_offsets = nullptr;
        int size = 0;
        DataState data_state = DataState::Invalid;

        DEVICE_RUNNABLE WkbTag get_tag(int index) const { return tags[index]; }

        // const pointer to start location to the index-th element
        // should be used when offsets are valid
        DEVICE_RUNNABLE const uint32_t* get_meta_ptr(int index) const {
            auto offset = meta_offsets[index];
            return metas + offset;
        }
        DEVICE_RUNNABLE const double* get_value_ptr(int index) const {
            auto offset = value_offsets[index];
            return values + offset;
        }

        // nonconst pointer to start location of the index-th element
        // should be used when offsets are valid
        DEVICE_RUNNABLE uint32_t* get_meta_ptr(int index) {
            auto offset = meta_offsets[index];
            return metas + offset;
        }
        DEVICE_RUNNABLE double* get_value_ptr(int index) {
            auto offset = value_offsets[index];
            return values + offset;
        }
    };

 private:
 public:
    // just a wrapper of unique_ptr<ctx_, dtor>
//    class GpuContextHolder {
//     public:
//        const GpuContext& get() const { return *ctx_; }
//        GpuContext& get() { return *ctx_; }
//        struct Deleter {
//            void operator()(GpuContext*);    // TODO
//        };
//        GpuContext* operator->() { return ctx_.operator->(); }
//
//     private:
//        std::unique_ptr<GpuContext, Deleter> ctx_;
//        explicit GpuContextHolder() : ctx_(new GpuContext) {}
//        friend class GeometryVector;
//    };
    static void GpuContextDeleter(GpuContext*);
    using GpuContextHolder = std::unique_ptr<GpuContext, DeleterWrapper<GpuContext, GpuContextDeleter>>;

    GpuContextHolder CreateReadGpuContext() const;    // TODO
    GeometryVector() = default;
    GpuVector<char> EncodeToWkb() const;    // TODO

    void WkbDecodeInitalize();
    // append single element
    void WkbDecodeAppend(const char* bin);
    void WkbDecodeFinalize();

    // STEP 1: Initialize vector with size of elements
    void OutputInitialize(int size);
    // STEP 2: Create gpu geometry context according to the vector for cuda,
    // where tags and offsets fields are uninitailized
    GpuContextHolder OutputCreateGpuContext();
    // STEP 3: Fill tags and offsets using CUDA Kernels

    // STEP 4: Exclusive scan offsets[0, n+1), where offsets[n] = 0
    // then copy info(tags and scanned offsets) back to GeometryVector 
    void OutputEvolveWith(GpuContext&);
    // STEP5: Copy data(metas and values) back to GeometryVector
    void OutputFinalizeWith(const GpuContext&);

    void clear();

    int size() const {
        auto tmp = tags_.size();
        assert(tmp <= std::numeric_limits<int>::max());
        return static_cast<int>(tmp);
    }

 private:
    // Currently, GpuVector contains host memory only
    // next goal should make it switchable between host and device memory.
    GpuVector<WkbTag> tags_;
    // Not including tags_, for faster access of WkbTags
    GpuVector<uint32_t> metas_;
    GpuVector<double> values_;
    // These two offsets fields contains
    //   FlatOffset => size of each element
    //   PrefixSumOffset => start location of each element
    GpuVector<int> meta_offsets_;
    GpuVector<int> value_offsets_;
    // This is the current state of above data containers and it companion GpuContext.
    // can only be used at assert statement for quick failure.
    // shouldn't be used to drive the state machine(e.g. switch statement)
    DataState data_state_ = DataState::Invalid;
};


using GpuContext = GeometryVector::GpuContext;

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
