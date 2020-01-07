#pragma once

#include "render/engine/operator/op_general_2d.h"


namespace zilliz {
namespace render {
namespace engine {


// Note: The size of colors.values must equel to the size of radius.values,
// and the order of them is one-to-one. (e.g.,radius of male group is 10, not 20)
//  ----------------- multi_color_2d VEGA JSON ------------------
//{
//    "width": 1024,
//    "height": 768,
//    "data": [
//        {
//            "name": "render_type",
//            "values": ["multi_color_circles_2d"]
//        },
//        {
//            "name": "colors",
//            "values": [
//                {
//                    "label" : "male",
//                    "color_r" : 255.0,
//                    "color_g" : 0.0,
//                    "color_b" : 0.0,
//                    "color_a" : 1.0
//                },
//                {
//                    "label" : "female",
//                    "color_r" : 0.0,
//                    "color_g" : 255.0,
//                    "color_b" : 0.0,
//                    "color_a" : 1.0
//                }
//            ]
//        },
//        {
//            "name": "radius",
//            "values": [10,20]
//        },
//        {
//            "name": "image_format",
//            "values": ["png"]
//        }
//    ]
//}
//  -----------------------------------------------------------------


class OpMultiColorCircles2D : public OpGeneral2D {
 public:
    DatasetPtr
    Render() override;
};


} // namespace engine
} // namespace render
} // namespace zilliz
