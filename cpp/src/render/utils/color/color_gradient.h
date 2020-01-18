#pragma once

#include <vector>


namespace zilliz {
namespace render {

class ColorGradient
{
 private:
    struct ColorPoint
    {
        float r,g,b;
        float val;
        ColorPoint(float red, float green, float blue, float value)
            : r(red), g(green), b(blue), val(value) {}
    };
    std::vector<ColorPoint> color;

 public:
    ColorGradient()  {  createDefaultHeatMapGradient();  }

    void createDefaultHeatMapGradient();

    void getColorAtValue(const float value, float &red, float &green, float &blue);
};

} // namespace render
} // namespace zilliz


