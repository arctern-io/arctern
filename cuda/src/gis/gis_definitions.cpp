#include <numeric>
#include "common/gpu_memory.h"
#include "gis/gis_definitions.h"


namespace zilliz::gis::cuda {

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
GeoWorkspaceConfigHolder
GeoWorkspaceConfigHolder::create(int max_buffer_per_meta, int max_buffer_per_value) {
    GeoWorkspaceConfigHolder holder;
    holder->max_buffer_per_meta = max_buffer_per_meta;
    holder->max_buffer_per_value = max_buffer_per_value;
    holder->meta_buffer = GpuAlloc<uint32_t>(holder->max_threads * max_buffer_per_meta);
    holder->value_buffer = GpuAlloc<double>(holder->max_threads * max_buffer_per_value);
}

void
GeoWorkspaceConfigHolder::destruct(GeoWorkspaceConfig* config) {
    GpuFree(config->meta_buffer);
    GpuFree(config->value_buffer);
}

}    // namespace zilliz::gis::cuda
