#pragma once

#include "render/engine/plan/plan/render_plan.h"
#include "render/engine/operator/dataset.h"
#include "render/engine/window/window.h"


namespace zilliz {
namespace render {
namespace engine {


class Operator {
 public:
    Operator() = default;

    virtual DatasetPtr
    Render() = 0;

 public:
    const RenderPlanPtr &
    plan() const { return plan_; }

    const DatasetPtr &
    input() { return input_; }

    const DataClientPtr
    data_client() { return data_client_; }

    const WindowPtr &
    window() const { return window_; }

    WindowPtr &
    mutable_window() { return window_; }

    void
    set_plan(RenderPlanPtr plan) { plan_ = plan; }

    void
    set_input(DatasetPtr input) { input_ = input; }

    void
    set_data_client(DataClientPtr data_client) { data_client_ = data_client; }

 private:
    RenderPlanPtr plan_;
    DatasetPtr input_;
    DataClientPtr data_client_;
    WindowPtr window_;
};

using OperatorPtr = std::shared_ptr<Operator>;

} // namespace engine
} // namespace render
} // namespace zilliz
