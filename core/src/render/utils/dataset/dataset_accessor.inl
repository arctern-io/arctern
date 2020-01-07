#pragma once

#include <cuda_runtime.h>

#include "chewie/error/error.h"
#include "zcommon/storage/storage_level.h"

#include "render/engine/common/error.h"
#include "render/engine/operator/dataset.h"
#include "render/engine/common/memory.h"


namespace zilliz {
namespace render {
namespace engine {

template<typename T>
std::shared_ptr<T>
DatasetAccessor::GetChunkData(const ChunkID &chunk_id) {

    chewie::rpc::Client::Data::GetResponsePtr get_rsps = nullptr;
    do {
        get_rsps = data_client_->Get(chunk_id, chewie::MM_Data_Partition_Id, false);
        if (get_rsps->status.ok()) {
            break;
        }
        if ((int32_t)get_rsps->status.error_code() == chewie::DATA_IS_NOT_SEALED) {
            RENDER_ENGINE_LOG_DEBUG << "GetData, but DATA_IS_NOT_SEALED is returned";
            usleep(2 * 1000);
        }
        else {
            std::string msg = "chunk data is not found. chunk_id = " + chunk_id.LogString();
            THROW_RENDER_ENGINE_ERROR(INPUT_NOT_FOUND, msg)
        }
    } while(1);

    T* chunk_data = (T*)get_rsps->buffer->data();
    size_t chunk_size = (size_t)get_rsps->size;

    if (get_rsps->partition_id == chewie::MM_Data_Partition_Id) {
        // The chunk data is already in main memory. We use it directly
        data_to_release_.emplace_back(chunk_id, chewie::MM_Data_Partition_Id);
        return std::shared_ptr<T>(chunk_data, [](T *) {});

    } else {
        // We need to copy the chunk data into main memory before use
        auto &mem_pool = MemManager::GetInstance().main_memory_pool();
        auto copied_data = std::shared_ptr<T>((T *)mem_pool.Malloc(chunk_size),
                                              [](T *p) { free(p); });
        CHECK_CUDA(cudaMemcpy(copied_data.get(), chunk_data, chunk_size, cudaMemcpyDeviceToHost))

        data_client_->Release(chunk_id, get_rsps->partition_id);
        return copied_data;
    }
}

template<typename T>
T *
DatasetAccessor::GetChunkDataForHeatmap(const ChunkID &chunk_id, int dev_index) {
    chewie::rpc::Client::Data::GetResponsePtr get_rsps = nullptr;
    do {
        chewie::DeviceID device_id = {zilliz::common::StorageLevel::kGPU, dev_index};
        chewie::PartitionID partition_id = chewie::ToPartitionID(device_id);
        get_rsps = data_client_->Get(chunk_id, partition_id, true);
        if (get_rsps->status.ok()) {
            break;
        }
        if ((int32_t)get_rsps->status.error_code() == chewie::DATA_IS_NOT_SEALED) {
            RENDER_ENGINE_LOG_DEBUG << "GetData, but DATA_IS_NOT_SEALED is returned";
            usleep(2 * 1000);
        }
        else {
            std::string msg = "chunk data is not found. chunk_id = " + chunk_id.LogString();
            THROW_RENDER_ENGINE_ERROR(INPUT_NOT_FOUND, msg)
        }
    } while(1);

    T* chunk_data = (T*)get_rsps->buffer->data();

    data_to_release_.emplace_back(chunk_id, get_rsps->partition_id);
    return chunk_data;
}


template<typename T>
void
DatasetAccessor::GetChunkData(const ChunkID &chunk_id, T *ptr, int64_t offset) {

    chewie::rpc::Client::Data::GetResponsePtr get_rsps = nullptr;
    do {
        get_rsps = data_client_->Get(chunk_id, chewie::MM_Data_Partition_Id, false);
        if (get_rsps->status.ok()) {
            break;
        }
        if ((int32_t)get_rsps->status.error_code() == chewie::DATA_IS_NOT_SEALED) {
            RENDER_ENGINE_LOG_DEBUG << "GetData, but DATA_IS_NOT_SEALED is returned";
            usleep(2 * 1000);
        }
        else {
            std::string msg = "chunk data is not found. chunk_id = " + chunk_id.LogString();
            THROW_RENDER_ENGINE_ERROR(INPUT_NOT_FOUND, msg)
        }
    } while(1);

    T* chunk_data = (T*)get_rsps->buffer->data();
    size_t chunk_size = (size_t)get_rsps->size;

    if (get_rsps->partition_id == chewie::MM_Data_Partition_Id) {
        memcpy(ptr + offset, chunk_data, chunk_size);
        data_client_->Release(chunk_id, chewie::MM_Data_Partition_Id);

    } else {
        CHECK_CUDA(cudaMemcpy(ptr + offset, chunk_data, chunk_size, cudaMemcpyDeviceToHost))
        data_client_->Release(chunk_id, get_rsps->partition_id);
    }
}


} // namespace engine
} // namespace render
} // namespace zilliz
