#include "render/engine/operator/op_heatmap_2d.h"
#include "render/engine/layer/layer_heatmap_2d.h"
#include "render/engine/common/log.h"
#include "render/engine/common/error.h"


namespace zilliz {
namespace render {
namespace engine {


DatasetPtr
OpHeatMap2D::Render() {

    Init();

    LayerHeatMap2D layer;
    layer.set_input(input());
    auto dataset_accessor = std::make_shared<DatasetAccessor>(data_client());
    layer.set_dataset_accessor(dataset_accessor);
    layer.set_plan_node(plan()->root_plan_node());
    layer.set_window_params(plan()->window_params());
    layer.Init();
    layer.Shader();
    memcpy(buffer_,layer.colors(),plan()->window_params().width() * plan()->window_params().height() * 4);
    window()->Terminate();
    return Output();
}


} // namespace engine
} // namespace render
} // namespace zilliz
