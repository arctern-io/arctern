#include <GLES3/gl3.h>

#include "render/engine/operator/op_circles_2d.h"
#include "render/engine/layer/layer_circles_2d.h"
#include "render/engine/common/log.h"
#include "render/engine/common/error.h"


namespace zilliz {
namespace render {
namespace engine {


DatasetPtr
OpCircles2D::Render() {

    Init();

    LayerCircles2D layer;
    layer.set_input(input());
    auto dataset_accessor = std::make_shared<DatasetAccessor>(data_client());
    layer.set_dataset_accessor(dataset_accessor);
    layer.set_plan_node(plan()->root_plan_node());
    layer.set_window_params(plan()->window_params());
    layer.Init();
    layer.Shader();
    layer.Render();

    Finalize();

    return Output();
}


} // namespace engine
} // namespace render
} // namespace zilliz