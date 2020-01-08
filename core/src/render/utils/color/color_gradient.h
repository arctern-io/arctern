#pragma once

#include <vector>


namespace zilliz {
namespace render {
namespace engine {


class ColorGradient
{
 private:
    struct ColorPoint
    {
        double r,g,b;
        double val;
        ColorPoint(double red, double green, double blue, double value)
            : r(red), g(green), b(blue), val(value) {}
    };
    std::vector<ColorPoint> color;

 public:
    ColorGradient()  {  createDefaultHeatMapGradient();  }

    void createDefaultHeatMapGradient();

    void getColorAtValue(const double value, double &red, double &green, double &blue);
};


} // namespace engine
} // namespace render
} // namespace zilliz


