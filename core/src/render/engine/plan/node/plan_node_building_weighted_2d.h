#pragma once

#include "render/engine/plan/plan/plan_node.h"
#include "render/engine/common/window_params.h"
#include "render/engine/plan/node/plan_node_weighted_color_circles_2d.h"
#include "render/utils/geo/geo_abstractor.h"


namespace zilliz {
namespace render {
namespace engine {

using CacheHint = zilliz::chewie::CacheHint;

class PlanNodeBuildingWeighted2D : public PlanNode {

 public:
    void
    Apply(Visitor &visitor) override { visitor.Visit(*this); }

    const ColorStyle
    color_style() const { return color_style_; }

    ColorStyle &
    mutable_color_style() { return color_style_; }

    double &
    count_start() { return count_start_; }

    double &
    count_end() { return count_end_; }

    void
    set_color_style(ColorStyle &color_style) { color_style_ = color_style; }

    void
    set_count_start(double &count_start) { count_start_ = count_start; }

    void
    set_count_end(double &count_end) { count_end_ = count_end; }

    void
    set_geo_type(GeoType &geo_type) { geo_type_ = geo_type; }

    void
    set_old_column_id(const ColumnID &old_column_id) {old_column_id_ = old_column_id; }

    const BoundingBox &
    bounding_box() const { return bounding_box_; }

    BoundingBox &
    mutable_bounding_box() { return bounding_box_; }

    const GeoType &
    geo_type() const { return geo_type_; }

    const ColumnID &
    old_column_id() const { return old_column_id_; }

 private:
    ColorStyle color_style_;
    GeoType geo_type_;
    double count_start_;
    double count_end_;
    BoundingBox bounding_box_;
    ColumnID old_column_id_;
};

using PlanNodeBuildingWeighted2DPtr = std::shared_ptr<PlanNodeBuildingWeighted2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
