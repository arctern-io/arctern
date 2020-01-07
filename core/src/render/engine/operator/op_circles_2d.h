#pragma once

#include "render/engine/operator/op_general_2d.h"


namespace zilliz {
namespace render {
namespace engine {


//  ---------- ------- single_color_2d VEGA JSON ------------------
//{
//    "width": 1024,
//    "height": 768,
//    "data": [
//        {
//            "name": "render_type",
//            "values": ["circles_2d"]
//        },
//        {
//            "name": "colors",
//            "values": [
//                {
//                    "color_r" : 1.0,
//                    "color_g" : 2.0,
//                    "color_b" : 3.0,
//                    "color_a" : 1.0
//                }
//            ]
//        },
//        {
//            "name": "radius",
//            "values": [10]
//        },
//        {
//            "name": "image_format",
//            "values": ["png"]
//        }
//    ]
//}
//  -----------------------------------------------------------------


class OpCircles2D : public OpGeneral2D {
 public:
    DatasetPtr
    Render() override;
};

using OpCircles2DPtr = std::shared_ptr<OpCircles2D>;

} // namespace engine
} // namespace render
} // namespace zilliz
