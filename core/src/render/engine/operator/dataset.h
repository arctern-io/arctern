#pragma once

#include "render/engine/common/table_id.h"
#include "zcommon/id/table_id.h"
#include "render/engine/store/obj_store.h"
#include "bulletin/accessor/accessor.h"
#include "bulletin/accessor/fragment.h"
#include "chewie/grpc/data_client.h"

namespace zilliz {
namespace render {
namespace engine {

struct Dataset {
 public:
    using DataID = zilliz::common::TableIDAttrEx;
    using ObjectBuffer = zilliz::chewie::Buffer;
    using ObjectBufferPtr = zilliz::chewie::BufferPtr;
    using DeviceID = zilliz::common::DeviceID;

    struct TableInfo {
        std::vector<ColumnField> columns;
        std::vector<FragmentField> fragments;
    };

    struct Data {
        DeviceID location;
        ObjectBufferPtr buffer;
    };

    using DataPtr = std::shared_ptr<Data>;

    using Meta = zilliz::bulletin::Accessor;
    using MetaPtr = zilliz::bulletin::AccessorPtr;


 public:
    std::map<TableID, TableInfo> table_info_;
    std::map<DataID, MetaPtr> meta_map_;
    std::map<DataID, DataPtr> data_map_;
};

using DatasetPtr = std::shared_ptr<Dataset>;
using FragmentMeta = zilliz::bulletin::FragmentBoard;
using FragmentMetaPtr = zilliz::bulletin::FragmentBoardPtr;
using DataClientPtr = std::shared_ptr<chewie::rpc::Client::Data>;

} // namespace engine
} // namespace render
} // namespace zilliz
