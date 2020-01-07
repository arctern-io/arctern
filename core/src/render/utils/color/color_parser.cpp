//#include <sstream>
#include "render/engine/common/error.h"
#include "render/utils/color/color_parser.h"


namespace zilliz {
namespace render {
namespace engine {

#define WHITE     0xFFFFFF;
#define YELLOW    0xFFFF00; //  :255,255,0
#define ORANGE    0xFF7D00; //  :255,125,0
#define PURPLE    0xFF00FF; //  :255,0,25
#define RED       0xFF0000; //  :255,0,0
#define CYANBLUE  0x00FFFF; //  :0,255,255
#define GREEN     0x00FF00; //  :0,255,0
#define BLUE      0x0000FF; //  :0,0,255
#define SKYBLUE   0xB4E7F5;

#define TRANSPARENCY 0.5f

void HexToRGB(int64_t color, CircleParams2D &circle_params_2d, ColorStyle color_style);

CircleParams2D
ColorParser::GetCircleParams(ColorStyle color_style, double ratio) {
    CircleParams2D circle_params_2d;

    switch (color_style) {
        case ColorStyle::kBlueToRed: {
            circle_params_2d.color.a = 1.0;
            circle_params_2d.color.r = ratio;
            circle_params_2d.color.g = 0.0;
            circle_params_2d.color.b = 1 - ratio;
            break;
        }
        case ColorStyle::kPurpleToYellow: {
            circle_params_2d.color.a = 1.0;
            circle_params_2d.color.r = 1.0;
            circle_params_2d.color.g = ratio;
            circle_params_2d.color.b = 1 - ratio;
            break;
        }
        case ColorStyle::kSkyBlueToWhite : {
            int64_t sky_blue = SKYBLUE;
            circle_params_2d.color.a = 1.0;
            circle_params_2d.color.r = ((255 - (sky_blue >> 16 & 0xff)) * ratio) / 255.0f;
            circle_params_2d.color.g = ((255 - (sky_blue >> 8 & 0xff)) * ratio) / 255.0f;
            circle_params_2d.color.b = ((255 - (sky_blue & 0xff)) * ratio) / 255.0f;
            break;
        }
        case ColorStyle::kRedTransParency: {
            int64_t red_lp = RED;
            circle_params_2d.color.a = ratio + TRANSPARENCY;

            HexToRGB(red_lp, circle_params_2d, color_style);
            break;
        }
        case ColorStyle::kBlueTransParency: {
            int64_t blue_lp = BLUE;
            circle_params_2d.color.a = ratio + TRANSPARENCY;

            HexToRGB(blue_lp, circle_params_2d, color_style);
            break;
        }
        case ColorStyle::kBlueGreenYellow: {
            circle_params_2d.color.r = (17.0f + (208.0f - 17.0f) * ratio) / 255.0f;
            circle_params_2d.color.g = (95.0f + (244.0f - 95.0f) * ratio) / 255.0f;
            circle_params_2d.color.b = (154.0f * (1.0f - ratio)) / 255.0f;
            circle_params_2d.color.a = 1.0;
            break;
        }
        case ColorStyle::kWhiteToBlue :{
            circle_params_2d.color.r = (226.0f + (17.0f - 226.0f) * ratio) / 255.0f;
            circle_params_2d.color.g = (226.0f + (95.0f - 226.0f) * ratio) / 255.0f;
            circle_params_2d.color.b = (226.0f + (154.0f - 226.0f) * ratio) / 255.0f;
            circle_params_2d.color.a = 1.0;
            break;
        }
        case ColorStyle::kGreenYellowRed: {
            circle_params_2d.color.r = (77.0f + (194.0f - 77.0f) * ratio) / 255.0f;
            circle_params_2d.color.g = (144.0f + (55.0f - 144.0f) * ratio) / 255.0f;
            circle_params_2d.color.b = (79.0f + (40.0f - 79.0f) * ratio) / 255.0f;
            circle_params_2d.color.a = 1.0;
            break;
        }
        case ColorStyle::kBlueWhiteRed: {
            circle_params_2d.color.r = (25.0f + (194.0f - 25.0f) * ratio) / 255.0f;
            circle_params_2d.color.g = (132.0f + (55.0f - 132.0f) * ratio) / 255.0f;
            circle_params_2d.color.b = (197.0f + (40.0f - 197.0f) * ratio) / 255.0f;
            circle_params_2d.color.a = 1.0;
            break;
        }
        default: {
            std::string msg = "cannot find color style";
            THROW_RENDER_ENGINE_ERROR(COLOR_STYLE_NOT_FOUND, msg)
            break;
        }
    }
    return circle_params_2d;
}

void HexToRGB(int64_t color, CircleParams2D &circle_params_2d, ColorStyle color_style) {
    switch (color_style) {
        case ColorStyle::kBlueToRed:
        case ColorStyle::kPurpleToYellow:
        case ColorStyle::kSkyBlueToWhite:
            circle_params_2d.color.r = (color >> 16 & 0xff) / 255.0f;
            circle_params_2d.color.g = (color >> 8 & 0xff) / 255.0f;
            circle_params_2d.color.b = (color & 0xff) / 255.0f;
            break;
        case ColorStyle::kBlueTransParency:
        case ColorStyle::kRedTransParency: {
            circle_params_2d.color.r = (color >> 16 & 0xff) / 255.0f;
            circle_params_2d.color.g = (color >> 8 & 0xff) / 255.0f;
            circle_params_2d.color.b = (color & 0xff) / 255.0f;
            break;
        }
        default: {
            std::string msg = "cannot find color style";
            THROW_RENDER_ENGINE_ERROR(COLOR_STYLE_NOT_FOUND, msg)
            break;
        }
    }
}

} // namespace engine
} // namespace render
} // namespace zilliz