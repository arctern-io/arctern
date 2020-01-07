#include "render/utils/dataset/dataset_accessor.h"
#include "render/engine/common/dyn_cast.h"

namespace zilliz {
namespace render {
namespace engine {

DatasetAccessor::DatasetAccessor(const DataClientPtr &data_client)
    : data_client_(data_client)
    , data_to_release_() {
}

DatasetAccessor::~DatasetAccessor() {
//    for (auto &ref : data_to_release_) {
//        data_client_->Release(ref.first, chewie::MM_Data_Partition_Id);
//    }
}

int64_t
DatasetAccessor::GetNumRows(const DatasetPtr &dataset, const FragmentID &fragment_id) {

    auto iter = dataset->meta_map_.find(fragment_id);

    if (iter == dataset->meta_map_.end()) {
        std::string msg = "fragment meta not found. fragment_id = " + fragment_id.LogString();
        THROW_RENDER_ENGINE_ERROR(INPUT_NOT_FOUND, msg)
    }

    return (dyn_cast<FragmentMeta>(dataset->meta_map_[fragment_id]))->num_rows();
}


std::vector<FragmentField>
DatasetAccessor::GetFragments(const DatasetPtr &dataset, const TableID &table_id) {

    auto iter = dataset->table_info_.find(table_id);

    if (iter == dataset->table_info_.end()) {
        std::string msg = "table info not found. table_id = " + table_id.LogString();
        THROW_RENDER_ENGINE_ERROR(INPUT_NOT_FOUND, msg)
    }

    return iter->second.fragments;
}


} // namespace engine
} // namespace render
} // namespace zilliz
