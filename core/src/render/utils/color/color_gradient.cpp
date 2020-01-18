#include "color_gradient.h"


namespace zilliz {
namespace render {

void ColorGradient::createDefaultHeatMapGradient()
{
    color.clear();
    color.push_back(ColorPoint(0, 0, 1,   0.0f));      // Blue.
    color.push_back(ColorPoint(0, 1, 1,   0.25f));     // Cyan.
    color.push_back(ColorPoint(0, 1, 0,   0.5f));      // Green.
    color.push_back(ColorPoint(1, 1, 0,   0.75f));     // Yellow.
    color.push_back(ColorPoint(1, 0, 0,   1.0f));      // Red.
}

void ColorGradient::getColorAtValue(const float value, float &red, float &green, float &blue)
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
            float value_diff    = (prev_color.val - curr_color.val);
            float fract_between = (value_diff==0) ? 0 : (value - curr_color.val) / value_diff;
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

} // namespace render
} // namespace zilliz