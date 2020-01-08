#pragma once

#include <memory>
#include <vector>
#include <map>
#include "render/engine/plan/prim/circle.h"
#include "render/engine/plan/plan/plan_node.h"
#include "render/engine/operator/dataset.h"
#include "render/engine/common/window_params.h"
#include "render/utils/dataset/dataset_accessor.h"

namespace zilliz {
namespace render {
namespace engine {


class Layer {
 public:
    Layer() = default;

    ~Layer() {
//        for (auto &ref : dataset_accessor()->data_to_release()) {
//            dataset_accessor()->data_client()->Release(ref.first, ref.second);
//        }
    }

    virtual void
    Init() = 0;

    virtual void
    Render() = 0;

    virtual void
    Shader() = 0;

    virtual void
    set_plan_node(PlanNodePtr plan_node) = 0;

 public:
    void
    set_input(DatasetPtr input) { input_ = input; }

    const DatasetPtr &
    input() const { return input_; }

    void
    set_dataset_accessor(DatasetAccessorPtr dataset_accessor) { dataset_accessor_ = dataset_accessor; }

    const DatasetAccessorPtr &
    dataset_accessor() const { return dataset_accessor_; }

 private:
    DatasetPtr input_;
    DatasetAccessorPtr dataset_accessor_;
};


using LayerPtr = std::shared_ptr<Layer>;


} // namespace engine
} // namespace render
} // namespace zilliz
