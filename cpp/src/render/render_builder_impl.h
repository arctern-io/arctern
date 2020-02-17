#pragma once
namespace zilliz {
namespace render {

std::pair<uint8_t*, int64_t>
pointmap(uint32_t* arr_x, uint32_t* arr_y, int64_t num, const std::string& conf) {

    PointMap point_map(arr_x, arr_y, num);

    VegaCircle2d vega_circle_2d(conf);
    point_map.mutable_point_vega() = vega_circle_2d;

    auto render = point_map.Render();
    auto ret_size = point_map.output_image_size();
    return std::make_pair(render, ret_size);
}

template<typename T>
std::pair<uint8_t*, int64_t>
heatmap(uint32_t* arr_x,
        uint32_t* arr_y,
        T* arr_c,
        int64_t num_vertices,
        const std::string& conf) {

    HeatMap<T> heat_map(arr_x, arr_y, arr_c, num_vertices);

    VegaHeatMap vega_heat_map(conf);
    heat_map.mutable_heatmap_vega() = vega_heat_map;

    auto render = heat_map.Render();
    auto ret_size = heat_map.output_image_size();
    return std::make_pair(render, ret_size);
}
}
}
