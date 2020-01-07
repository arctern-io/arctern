#pragma once

#include "render/engine/operator/operator.h"


namespace zilliz {
namespace render {
namespace engine {

class OpCursorInter : public Operator {
 public:
    DatasetPtr
    Render() override;

 private:

    DatasetPtr
    Output();

 private:
    std::string building_wkt_;
};

using OpCursorInterPtr = std::shared_ptr<OpCursorInter>;

} // namespace engine
} // namespace render
} // namespace zilliz