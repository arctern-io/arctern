#pragma once

#include "render/engine/operator/op_general_2d.h"


namespace zilliz {
namespace render {
namespace engine {


//  ------------------ weighted_pointsize_2d VEGA JSON ------------------
//{
//    "width": 1024,
//    "height": 768,
//    "data": [
//        {
//            "name": "render_type",
//            "values": ["weighted_pointsize_circles_2d"]
//        },
//        {
//            "name": "colors",
//            "values": [
//                {
//                    "color_r" : 255.0,
//                    "color_g" : 0.0,
//                    "color_b" : 0.0,
//                    "color_a" : 1.0
//                }
//        },
//        {
//            "name": "image_format",
//            "values": ["png"]
//        }
//    ]
//}
//
//
//  -----------------------------------------------------------------


class OpWeightedPointSizeCircles2D : public OpGeneral2D {
 public:
    DatasetPtr
    Render() final;
};


} // namespace engine
} // namespace render
} // namespace zilliz
