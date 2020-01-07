#include "render/engine/operator/op_building_weighted_2d.h"
#include "render/engine/layer/layer_building_weighted_2d.h"


namespace zilliz {
namespace render {
namespace engine {


DatasetPtr
OpBuildingWeighted2D::Render() {

    Init();

    auto dataset_accessor = std::make_shared<DatasetAccessor>(data_client());
    auto measure_column_type = plan()->root_plan_node()->data_param_type()[1];

    auto get_layer = [&] (auto &layer) {
        layer.set_input(input());
        layer.set_dataset_accessor(dataset_accessor);
        layer.set_plan_node(plan()->root_plan_node());
        layer.set_window_params(window()->window_params());
        layer.Init();
        layer.Render();
    };

    switch (measure_column_type) {
        case ValueType::kValInt8: {
            LayerBuildingWeighted2D<int8_t> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValInt16: {
            LayerBuildingWeighted2D<int16_t> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValInt32: {
            LayerBuildingWeighted2D<int32_t> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValInt64: {
            LayerBuildingWeighted2D<int64_t> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValUInt8: {
            LayerBuildingWeighted2D<uint8_t> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValUInt16: {
            LayerBuildingWeighted2D<uint16_t> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValUInt32: {
            LayerBuildingWeighted2D<uint32_t> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValUInt64: {
            LayerBuildingWeighted2D<uint64_t> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValFloat: {
            LayerBuildingWeighted2D<float> layer;
            get_layer(layer);
            break;
        }
        case ValueType::kValDouble: {
            LayerBuildingWeighted2D<double> layer;
            get_layer(layer);
            break;
        }
        default:std::string msg = "cannot find value type";
            THROW_RENDER_ENGINE_ERROR(VALUE_TYPE_NOT_FOUND, msg);
    }

    Finalize();

    return Output();
}


} // namespace engine
} // namespace render
} // namespace zilliz