#pragma once

#include "zstring/StringEngineOwner.h"

#include "render/engine/plan/node/plan_node_building_weighted_2d.h"
#include "render/engine/layer/layer.h"
#include "render/engine/common/dyn_cast.h"
#include "render/utils/geo/geo_building_reader.h"


namespace zilliz {
namespace render {
namespace engine {

template<typename T>
class LayerBuildingWeighted2D : public Layer {
 public:
    LayerBuildingWeighted2D();

 public:
    void
    Render() final;

    void
    Init() final;

    void
    Shader() final;

    const PlanNodeBuildingWeighted2DPtr &
    plan_node() const { return plan_node_; }

    void
    set_plan_node(PlanNodePtr plan_node) override {
        plan_node_ = dyn_cast<PlanNodeBuildingWeighted2D>(plan_node);
    }

    void
    set_window_params(WindowParams window_params) { window_params_ = window_params; }

 private:

    void
    SetVerticesAndColors();

    void
    TransForm();

    zstring::StringEngineOwnerPtr
    GetStringEngineOwner(const ColumnID &column_id);

 private:
    PlanNodeBuildingWeighted2DPtr plan_node_;
    WindowParams window_params_;

    std::vector<int64_t> point_list_;
    std::shared_ptr<T> count_;

    size_t num_buildings_;

    std::vector<std::vector<int>> buildings_x_;
    std::vector<std::vector<int>> buildings_y_;

    std::vector<float> colors_;
};


} // namespace engine
} // namespace render
} // namespace zilliz