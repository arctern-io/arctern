#pragma once

#include "render/engine/operator/op_general_2d.h"


namespace zilliz {
namespace render {
namespace engine {


//  ------------------ heatmap_2d VEGA JSON ------------------
//{
//    "width": 1024,
//    "height": 768,
//    "data": [
//        {
//            "name": "render_type",
//            "values": ["heatmap_2d"]
//        },
//        {
//            "name": "color_style",
//            "values": ["style":"blue_to_red"]
//        },
//        {
//            "name": "image_format",
//            "values": ["png"]
//        }
//    ]
//}
//
// @data.color_style possible values:
//  "blue_to_red"  blue for small count, red for large count, similar to heat map
//  "red_transparency" larger the count means more solid circle color
//
//  -----------------------------------------------------------------


class OpHeatMap2D : public OpGeneral2D {
 public:
    DatasetPtr
    Render() final;
};


} // namespace engine
} // namespace render
} // namespace zilliz
