#pragma once

#include "rapidjson/document.h"

#include "render/engine/plan/node/plan_node_icon_2d.h"
#include "render/engine/plan/node/plan_node_circles_2d.h"
#include "render/engine/plan/node/plan_node_multi_color_circles_2d.h"
#include "render/engine/plan/node/plan_node_weighted_color_circles_2d.h"
#include "render/engine/plan/node/plan_node_weighted_pointsize_circles_2d.h"
#include "render/engine/plan/node/plan_node_weighted_circles_2d.h"
#include "render/engine/plan/node/plan_node_heatmap_2d.h"
#include "render/engine/plan/plan/render_plan.h"


namespace zilliz {
namespace render {
namespace engine {


class VegaParser : public PlanNode::Visitor {
 public:
    using Value = rapidjson::Value;

 public:
    RenderPlanPtr
    GetPlan(const std::string &query);

 public: // PlanNode::Visitor functions
    void
    Visit(PlanNodeIcon2D &plan_node) override;

    void
    Visit(PlanNodeCircle2D &plan_node) override;

    void
    Visit(PlanNodeMultiColorCircle2D &plan_node) override;

    void
    Visit(PlanNodeWeightedColorCircle2D &plan_node) override;

    void
    Visit(PlanNodeBuildingWeighted2D &plan_node) override;

    void
    Visit(PlanNodeWeightedPointSizeCircle2D &plan_node) override;

    void
    Visit(PlanNodeWeightedCircle2D &plan_node) override;

    void
    Visit(PlanNodeHeatMap2D &plan_node) override;

    void
    Visit(PlanNodeCursorInter &plan_node) override;

 private:
    bool
    VegaJsonFormatCheck(const std::string &query);

    static bool
    JsonLabelCheck(Value &value, const std::string& label);

    static bool
    JsonSizeCheck(Value &value, const std::string& label, size_t size);

    bool
    WindowParamsCheck();

    bool
    RenderTypeCheck(int i);

    bool
    RadiusCheck(int i);

    bool
    ImageFormatCheck(int i);

    bool
    SingleColorCheck();

    bool
    MultiColorCheck();

    bool
    WeightedColorStyleCheck();

    bool
    BuildingColorStyleCheck();

    bool
    WeightedPointSizeStyleCheck();

    bool
    HeatmapStyleCheck();

    bool
    HasColorCheck();

    bool
    BoundBoxCheck();

 private:
    WindowParams
    GetWindowParamsFromJson();

    ImageFormatPtr
    GetImageFormatFromJson();

    PlanNodePtr
    GetPlanNodesFromJson();

    CircleParams2D
    GetCircleParams2DFromJson(int i);

 private:
    void
    SetCommonValueFromJson();

    void
    SetColorValueFromJson();

    void
    SetBoundBoxFromJson();

    void
    SetGeoTypeFromJson();

 private:
    rapidjson::Document document_;
    Value data_;
    Value radius_;
    Value colors_;
    Value map_scale_ratio_;
    Value color_style_;
    Value image_format_;
    Value render_layers_;
    Value bound_box_;
    Value geo_type_;
};


} // namespace engine
} // namespace render
} // namespace zilliz
