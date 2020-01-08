#include "render/utils/color/color_gradient.h"


namespace zilliz {
namespace render {
namespace engine {

void ColorGradient::createDefaultHeatMapGradient()
{
    color.clear();
    color.push_back(ColorPoint(0, 0, 1,   0.0));      // Blue.
    color.push_back(ColorPoint(0, 1, 1,   0.25));     // Cyan.
    color.push_back(ColorPoint(0, 1, 0,   0.5));      // Green.
    color.push_back(ColorPoint(1, 1, 0,   0.75));     // Yellow.
    color.push_back(ColorPoint(1, 0, 0,   1.0));      // Red.
}

void ColorGradient::getColorAtValue(const double value, double &red, double &green, double &blue)
{
    if(color.size()==0)
        return;

    for(unsigned int i=0; i<color.size(); i++)
    {
        ColorPoint &curr_color = color[i];
        if(value < curr_color.val)
        {
            int index = (i-1) > 0 ? i-1: 0;
            ColorPoint &prev_color  = color[index];
            double value_diff    = (prev_color.val - curr_color.val);
            double fract_between = (value_diff==0) ? 0 : (value - curr_color.val) / value_diff;
            red   = (prev_color.r - curr_color.r)*fract_between + curr_color.r;
            green = (prev_color.g - curr_color.g)*fract_between + curr_color.g;
            blue  = (prev_color.b - curr_color.b)*fract_between + curr_color.b;
            return;
        }
    }
    red   = color.back().r;
    green = color.back().g;
    blue  = color.back().b;
    return;
}

} // namespace engine
} // namespace render
} // namespace zilliz