#include <GL/gl.h>

#include "zstring/DictStringEngineAgent.h"
#include "zstring/HashStringEngineAgent.h"
#include "zstring/ShortStringEngineAgent.h"

#include "zcommon/util/string_builder.h"

#include "render/utils/color/color_parser.h"
#include "render/utils/dataset/dataset_accessor.h"
#include "render/utils/geo/building_cache.h"

#include "render/engine/layer/layer_building_weighted_2d.h"


namespace zilliz {
namespace render {
namespace engine {

template
class LayerBuildingWeighted2D<int8_t>;

template
class LayerBuildingWeighted2D<int16_t>;

template
class LayerBuildingWeighted2D<int32_t>;

template
class LayerBuildingWeighted2D<int64_t>;

template
class LayerBuildingWeighted2D<uint8_t>;

template
class LayerBuildingWeighted2D<uint16_t>;

template
class LayerBuildingWeighted2D<uint32_t>;

template
class LayerBuildingWeighted2D<uint64_t>;

template
class LayerBuildingWeighted2D<float>;

template
class LayerBuildingWeighted2D<double>;

template<typename T>
LayerBuildingWeighted2D<T>::LayerBuildingWeighted2D()
    : num_buildings_(0) {}

template<typename T>
void
LayerBuildingWeighted2D<T>::Shader() {}


template<typename T>
void
LayerBuildingWeighted2D<T>::Init() {
    SetVerticesAndColors();
    TransForm();
}


template<typename T>
zstring::StringEngineOwnerPtr
LayerBuildingWeighted2D<T>::GetStringEngineOwner(const ColumnID &column_id) {
    auto string_type = common::GetIntermediateIdType(column_id);
    switch (string_type) {
        case common::TextGroupKeygenType::kTreeKey: {
            return sql::agent::DictStringEngineAgent::GetInstance().GetGroup(
                common::GetStringGroupID(column_id.db_id(),
                                         column_id.table_id(),
                                         column_id.column_id()));
        }
        case common::TextGroupKeygenType::kHashKey: {
            return sql::agent::HashStringEngineAgent::GetInstance().GetGroup(
                common::GetStringGroupID(column_id.db_id(),
                                         column_id.table_id(),
                                         column_id.column_id()));
        }
        case common::TextGroupKeygenType::kShortString: {
            return sql::agent::ShortStringEngineAgent::Generate();
        }
        default: {
            std::string msg = "Unknown string type '" + std::to_string(string_type) + "'";
            THROW_RENDER_ENGINE_ERROR(UNKNOWN_STRING_TYPE, msg)
        }
    }
}


template<typename T>
void
LayerBuildingWeighted2D<T>::Render() {
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ZERO);

    for (int i = 0; i < num_buildings_; i++) {
        glColor4f(colors_[i * 4], colors_[i * 4 + 1], colors_[i * 4 + 2], colors_[i * 4 + 3]);
        glBegin(GL_POLYGON);
        for (int j = 0; j < buildings_x_[i].size(); j++) {
            glVertex2d(buildings_x_[i][j], buildings_y_[i][j]);
        }
        glEnd();
    }
}


template<typename T>
void
LayerBuildingWeighted2D<T>::TransForm() {

    auto owner = GetStringEngineOwner(plan_node_->old_column_id());

    buildings_x_.resize(num_buildings_);
    buildings_y_.resize(num_buildings_);

    auto bounding_box = plan_node_->bounding_box();

    auto x_left = bounding_box.longitude_left * 111319.490778;
    auto x_right = bounding_box.longitude_right * 111319.490778;

    auto y_left = 6378136.99911 * log(tan(.00872664626 * bounding_box.latitude_left + .785398163397));
    auto y_right = 6378136.99911 * log(tan(.00872664626 * bounding_box.latitude_right + .785398163397));

    auto width = window_params_.width();
    auto height = window_params_.height();

    BuildingCache::GetInstance().mutable_building_buffer().clear();

    for (int i = 0; i < num_buildings_; i++) {

        std::string point_list_string;
        owner->GetStringById(point_list_[i], point_list_string);
        if (point_list_string.empty()) {
            continue;
        }

        OGRGeometry *geometry;
        OGRGeometryFactory::createFromWkt(point_list_string.c_str(), nullptr, &geometry);

        auto type = geometry->getGeometryType();

        if (type == OGRwkbGeometryType::wkbPolygon) {

            auto ring = geometry->toPolygon()->getExteriorRing();

            BuildingCache::GetInstance().mutable_building_buffer().emplace_back(std::make_pair(point_list_string, *ring));

            auto ring_size = ring->getNumPoints();
            buildings_x_[i].resize(ring_size);
            buildings_y_[i].resize(ring_size);
            for (int j = 0; j < ring_size; j++) {

                double x_pos = ring->getX(j) * 111319.490778;
                int ret_x = (int) (((x_pos - x_left) / (x_right - x_left)) * width - 1E-9);
                buildings_x_[i][j] = ret_x;

                double y_pos = 6378136.99911 * log(tan(.00872664626 * ring->getY(j) + .785398163397));
                int ret_y = (int) (((y_pos - y_left) / (y_right - y_left)) * height - 1E-9);
                buildings_y_[i][j] = ret_y;
            }

        } else {
            RENDER_ENGINE_LOG_ERROR << "Unknown geometry type.";
        }
    }
}

template<typename T>
void
LayerBuildingWeighted2D<T>::SetVerticesAndColors() {

    auto &data_params = plan_node_->mutable_data_params();
    auto list_id = data_params[0];
    auto count_id = data_params[1];

    auto table_id = list_id;
    table_id.truncate_to_table_id();
    auto fragments_field = DatasetAccessor::GetFragments(input(), table_id);

    std::vector<int64_t> num_rows;
    num_rows.clear();
    for (size_t i = 0; i < fragments_field.size(); i++) {
        list_id.set_fragment_field(fragments_field[i]);
        list_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kMeta);
        auto fragment_id = list_id;
        fragment_id.truncate_to_fragment_id();
        int64_t num_row = DatasetAccessor::GetNumRows(input(), fragment_id);
        num_rows.emplace_back(num_row);
        num_buildings_ += num_row;
    }

    if (num_buildings_ <= 0) {
        RENDER_ENGINE_LOG_WARNING << "There is no building to show.";
        return;
    }

    colors_.resize(num_buildings_ * 4);
    point_list_.resize(num_buildings_);

    size_t fragment_offset = 0;
    size_t c_offset = 0;
    auto count_start = plan_node()->count_start();
    auto count_end = plan_node()->count_end();
    auto count_range = count_end - count_start;

    for (size_t i = 0; i < fragments_field.size(); i++) {

        list_id.set_fragment_field(fragments_field[i]);
        count_id.set_fragment_field(fragments_field[i]);

        list_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        count_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);

        dataset_accessor()->template GetChunkData<int64_t>(list_id, &point_list_[0], fragment_offset);
        fragment_offset += num_rows[i];

        count_ = dataset_accessor()->template GetChunkData<T>(count_id);

        for (auto j = 0; j < num_rows[i]; j++) {
            auto color_style = plan_node()->color_style();
            auto count = count_.get()[j] >= count_start ? count_.get()[j] : count_start;
            count = count_.get()[j] <= count_end ? count : count_end;
            auto ratio = (count - count_start) / count_range;
            auto circle_params_2d = ColorParser::GetCircleParams(color_style, ratio);
            colors_[c_offset++] = circle_params_2d.color.r;
            colors_[c_offset++] = circle_params_2d.color.g;
            colors_[c_offset++] = circle_params_2d.color.b;
            colors_[c_offset++] = circle_params_2d.color.a;
        }
    }
}


} // namespace engine
} // namespace render
} // namespace zilliz

