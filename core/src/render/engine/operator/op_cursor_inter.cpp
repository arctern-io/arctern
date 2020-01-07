#include "render/engine/common/dyn_cast.h"
#include "render/engine/operator/op_cursor_inter.h"
#include "render/utils/geo/building_cache.h"

#include "zcommon/config/megawise_config.h"


namespace zilliz {
namespace render {
namespace engine {


DatasetPtr
OpCursorInter::Render() {

    auto plan_node = dyn_cast<PlanNodeCursorInter>(plan()->mutable_root_plan_node());
    auto &cursor_x = plan_node->cursor_position().first;
    auto &cursor_y = plan_node->cursor_position().second;
    auto &building_buffer = BuildingCache::GetInstance().building_buffer();

    OGREnvelope envelope;

    for (const auto &building : building_buffer) {
        building.second.getEnvelope(&envelope);
        if (cursor_x < envelope.MinX ||
            cursor_x > envelope.MaxX ||
            cursor_y < envelope.MinY ||
            cursor_y > envelope.MaxY) {
            continue;
        }

        OGRPoint point(cursor_x, cursor_y);
        if (building.second.isPointInRing(&point)) {
            building_wkt_ = building.first;
            break;
        }
    }

    return Output();
}


DatasetPtr
OpCursorInter::Output() {

    auto output = std::make_shared<Dataset>();
    auto &plan_node = plan()->root_plan_node();

    if (building_wkt_.length() <= 0) {
        RENDER_ENGINE_LOG_WARNING << "null building wkt.";
        return output;
    }

    TableID output_id(plan_node->output_id().db_id(),
                      plan_node->output_id().table_id());
    output_id.set_attr_type(TableID::AttrType::kMeta);

    // set table info
    // we have only one output string, such that num_fragment = 1, num_column = 1
    auto &table_info = output->table_info_[output_id];
    table_info.fragments.push_back(0);
    table_info.columns.push_back(0);

    output_id.set_column_id(common::TableID::kInvalidColumnField);
    output_id.set_fragment_id(0);
    // set fragment info
    // we have only one output string, such that num_rows = 1.
    zilliz::bulletin::FragmentBoardPtr
        fragment_board = std::make_shared<zilliz::bulletin::FragmentBoard>(output_id);
    fragment_board->set_num_rows(1);
    output->meta_map_[output_id] = std::static_pointer_cast<Dataset::Meta>(fragment_board);

    output_id.set_column_id(0);
    output_id.set_attr_type(TableID::AttrType::kData);

    auto write_image = common::megawise::DevCfg::render_engine::write_image();
    if (write_image) {
        std::cout << "wkt:" << building_wkt_ << std::endl;
    }

    int64_t string_length = building_wkt_.length();

    // 1. copy building wkt to dataset
    auto create_resp_buffer = data_client()->Create(output_id,
                                             chewie::MM_Data_Partition_Id,
                                             building_wkt_.length() + 36,
                                             chewie::CacheHint::fNotDroppable);

    auto buffer_pointer = (unsigned char *)create_resp_buffer->buffer->mutable_data();
    if (buffer_pointer == nullptr) {
        THROW_RENDER_ENGINE_ERROR(CREATE_RESPONSE_DATA_NULL, "chewie buffer data created is null.")
    }
    std::memcpy((unsigned char *) (create_resp_buffer->buffer->data()), &output_id, 20);
    *((int64_t *) (buffer_pointer + 20)) = string_length;
    *((int64_t *) (buffer_pointer + 28)) = 1;
    std::memcpy(buffer_pointer + 36, building_wkt_.c_str(), string_length);

    data_client()->Seal(output_id, chewie::MM_Data_Partition_Id);
    data_client()->Release(output_id, chewie::MM_Data_Partition_Id);

    // 2. send wkt length to chewie
    output_id.set_attr_type(TableID::AttrType::kDataOffset);
    auto create_offset_resp = data_client()->Create(output_id,
                                                         chewie::MM_Data_Partition_Id,
                                                         24,
                                                         chewie::CacheHint::fNotDroppable);
    auto offset_pointer = (int64_t *)create_offset_resp->buffer->mutable_data();
    if (offset_pointer == nullptr) {
        THROW_RENDER_ENGINE_ERROR(CREATE_RESPONSE_DATA_NULL, "chewie buffer data created is null.")
    }
    *offset_pointer++ = 1;
    *offset_pointer++ = 0;
    *offset_pointer++ = string_length;
    data_client()->Seal(output_id, chewie::MM_Data_Partition_Id);
    data_client()->Release(output_id, chewie::MM_Data_Partition_Id);

    return output;
}


} // namespace engine
} // namespace render
} // namespace zilliz