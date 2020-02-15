#include <numeric>
#include "common/gpu_memory.h"
#include "gis/gis_definitions.h"
#include <thrust/scan.h>


namespace zilliz {
namespace gis {
namespace cuda {

GeometryVector::GeoContextHolder
GeometryVector::CreateReadGeoContext() const {
    assert(data_state_ == DataState::PrefixSumOffset_FullData);

    GeometryVector::GeoContextHolder holder;
    static_assert(std::is_same<GpuVector<int>, vector<int>>::value,
                  "here use vector now");
    auto size = tags_.size();    // size_ of elements
    assert(size + 1 == meta_offsets_.size());
    assert(size + 1 == value_offsets_.size());
    assert(meta_offsets_[size] == metas_.size());
    assert(value_offsets_[size] == values_.size());
    holder->tags = GpuAllocAndCopy(tags_.data(), tags_.size());
    holder->metas = GpuAllocAndCopy(metas_.data(), metas_.size());
    holder->values = GpuAllocAndCopy(values_.data(), values_.size());
    holder->meta_offsets = GpuAllocAndCopy(meta_offsets_.data(), meta_offsets_.size());
    holder->value_offsets = GpuAllocAndCopy(value_offsets_.data(), value_offsets_.size());
    holder->size = tags_.size();
    holder->data_state = data_state_;
    return holder;
}

void
GeometryVector::GeoContextHolder::Deleter::operator()(GeoContext* ptr) {
    if (!ptr) {
        return;
    }
    GpuFree(ptr->tags);
    GpuFree(ptr->metas);
    GpuFree(ptr->values);
    GpuFree(ptr->meta_offsets);
    GpuFree(ptr->value_offsets);
    ptr->size = 0;
    ptr->data_state = DataState::Invalid;
}

GeoWorkspaceHolder
GeoWorkspaceHolder::create(int max_buffer_per_meta, int max_buffer_per_value) {
    GeoWorkspaceHolder holder;
    holder->max_buffer_per_meta = max_buffer_per_meta;
    holder->max_buffer_per_value = max_buffer_per_value;
    holder->meta_buffers = GpuAlloc<uint32_t>(holder->max_threads * max_buffer_per_meta);
    holder->value_buffers = GpuAlloc<double>(holder->max_threads * max_buffer_per_value);
    return holder;
}

void
GeoWorkspaceHolder::destruct(GeoWorkspace* ptr) {
    GpuFree(ptr->meta_buffers);
    GpuFree(ptr->value_buffers);
    ptr->max_buffer_per_value = 0;
    ptr->max_buffer_per_meta = 0;
}

void
GeometryVector::OutputInitialize(int size) {
    tags_.resize(size);
    meta_offsets_.resize(size + 1);
    value_offsets_.resize(size + 1);
    data_state_ = DataState::FlatOffset_EmptyInfo;
}

auto
GeometryVector::OutputCreateGeoContext() -> GeoContextHolder {
    assert(data_state_ == DataState::FlatOffset_EmptyInfo);
    GeoContextHolder holder;
    auto size = tags_.size();    // size_ of elements
    assert(size + 1 == meta_offsets_.size());
    assert(size + 1 == value_offsets_.size());
    assert(meta_offsets_[size] == metas_.size());
    assert(value_offsets_[size] == values_.size());
    holder->size = size;
    holder->tags = GpuAlloc<WkbTag>(tags_.size());
    holder->meta_offsets = GpuAlloc<int>(meta_offsets_.size());
    holder->value_offsets = GpuAlloc<int>(value_offsets_.size());
    holder->data_state = data_state_;
    assert(holder->metas == nullptr);
    assert(holder->values == nullptr);
    return holder;
}

// scan operation
// memory allocation for metas/values
// copy useful data back to CPU
void
GeometryVector::OutputEvolveWith(GeoContext& ctx) {
    assert(data_state_ == DataState::FlatOffset_EmptyInfo);
    assert(ctx.data_state == DataState::FlatOffset_FullInfo);
    assert(tags_.size() == ctx.size);
    auto size = ctx.size;
    GpuMemcpy(tags_.data(), ctx.tags, size);
    auto scan_at = [=](int* gpu_addr, int* cpu_addr) {
        int zero = 0;
        GpuMemcpy(gpu_addr + size, &zero, 1);
        thrust::exclusive_scan(
            thrust::cuda::par, gpu_addr, gpu_addr + size + 1, gpu_addr);
        GpuMemcpy(cpu_addr, gpu_addr, size + 1);

        int result;
        GpuMemcpy(&result, gpu_addr + size, 1);
        return result;
    };

    auto meta_size = scan_at(ctx.meta_offsets, meta_offsets_.data());
    auto value_size = scan_at(ctx.value_offsets, meta_offsets_.data());

    metas_.resize(meta_size);
    values_.resize(value_size);
    data_state_ = DataState::PrefixSumOffset_EmptyData;

    ctx.metas = GpuAlloc<uint32_t>(meta_size);
    ctx.values = GpuAlloc<double>(value_size);
    ctx.data_state = DataState::PrefixSumOffset_EmptyData;
}

void
GeometryVector::OutputFinalizeWith(const GeoContext& ctx) {
    assert(ctx.data_state == DataState::PrefixSumOffset_FullData);
    assert(data_state_ == DataState::PrefixSumOffset_EmptyData);
    GpuMemcpy(metas_.data(), ctx.metas, metas_.size());
    GpuMemcpy(values_.data(), ctx.values, values_.size());
    data_state_ = DataState::PrefixSumOffset_FullData;
}
}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
