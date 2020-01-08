#pragma once

#pragma once

#include "render/engine/operator/dataset.h"


namespace zilliz {
namespace render {
namespace engine {


class DatasetAccessor {
 public:
    DatasetAccessor(const DataClientPtr &data_client);
    ~DatasetAccessor();

    template<typename T>
    std::shared_ptr<T>
    GetChunkData(const ChunkID &chunk_id);

    template<typename T>
    void
    GetChunkData(const ChunkID &chunk_id, T *ptr, int64_t offset);

    template<typename T>
    T *
    GetChunkDataForHeatmap(const ChunkID &chunk_id, int dev_index);

    static int64_t
    GetNumRows(const DatasetPtr &dataset, const FragmentID &fragment_id);

    static std::vector<FragmentField>
    GetFragments(const DatasetPtr &dataset, const TableID &table_id);

    void
    set_data_client(const DataClientPtr &data_client) { data_client_ = data_client; }

    const DataClientPtr &
    data_client() { return data_client_; }

    const std::vector<std::pair<ChunkID, chewie::PartitionID> > &
    data_to_release() const { return data_to_release_; }

 private:
    DataClientPtr data_client_;
    std::vector<std::pair<ChunkID, chewie::PartitionID> > data_to_release_;
};

using DatasetAccessorPtr = std::shared_ptr<DatasetAccessor>;


} // namespace engine
} // namespace render
} // namespace zilliz


#include "render/utils/dataset/dataset_accessor.inl"
