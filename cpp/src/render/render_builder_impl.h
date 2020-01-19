#pragma once
namespace zilliz {
namespace render {

std::pair<uint8_t*, int64_t>
pointmap(uint32_t* arr_x, uint32_t* arr_y, int64_t num) {
    PointMap point_map(arr_x, arr_y, num);
    std::string vega = "{\n"
                       "  \"width\": 300,\n"
                       "  \"height\": 200,\n"
                       "  \"description\": \"circle_2d\",\n"
                       "  \"data\": [\n"
                       "    {\n"
                       "      \"name\": \"data\",\n"
                       "      \"url\": \"data/data.csv\"\n"
                       "    }\n"
                       "  ],\n"
                       "  \"scales\": [\n"
                       "    {\n"
                       "      \"name\": \"x\",\n"
                       "      \"type\": \"linear\",\n"
                       "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
                       "    },\n"
                       "    {\n"
                       "      \"name\": \"y\",\n"
                       "      \"type\": \"linear\",\n"
                       "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
                       "    }\n"
                       "  ],\n"
                       "  \"marks\": [\n"
                       "    {\n"
                       "      \"encode\": {\n"
                       "        \"enter\": {\n"
                       "          \"shape\": {\"value\": \"circle\"},\n"
                       "          \"stroke\": {\"value\": \"#ff0000\"},\n"
                       "          \"strokeWidth\": {\"value\": 30},\n"
                       "          \"opacity\": {\"value\": 0.5}\n"
                       "        }\n"
                       "      }\n"
                       "    }\n"
                       "  ]\n"
                       "}";
    VegaCircle2d vega_circle_2d(vega);
    point_map.mutable_point_vega() = vega_circle_2d;

    return std::make_pair(point_map.Render(), point_map.output_image_size());
}

template<typename T>
std::pair<uint8_t*, int64_t>
heatmap(uint32_t* arr_x,
        uint32_t* arr_y,
        T* arr_c,
        int64_t num_vertices) {
    HeatMap<T> heat_map(arr_x, arr_y, arr_c, num_vertices);
    std::string vega = "{\n"
                       "  \"width\": 300,\n"
                       "  \"height\": 200,\n"
                       "  \"description\": \"circle_2d\",\n"
                       "  \"data\": [\n"
                       "    {\n"
                       "      \"name\": \"data\",\n"
                       "      \"url\": \"data/data.csv\"\n"
                       "    }\n"
                       "  ],\n"
                       "  \"scales\": [\n"
                       "    {\n"
                       "      \"name\": \"x\",\n"
                       "      \"type\": \"linear\",\n"
                       "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
                       "    },\n"
                       "    {\n"
                       "      \"name\": \"y\",\n"
                       "      \"type\": \"linear\",\n"
                       "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
                       "    }\n"
                       "  ],\n"
                       "  \"marks\": [\n"
                       "    {\n"
                       "      \"encode\": {\n"
                       "        \"enter\": {\n"
                       "          \"map_scale\": {\"value\": 10}\n"
                       "        }\n"
                       "      }\n"
                       "    }\n"
                       "  ]\n"
                       "}";
    VegaHeatMap vega_heat_map(vega);
    heat_map.mutable_heatmap_vega() = vega_heat_map;

    return std::make_pair(heat_map.Render(), heat_map.output_image_size());
}
}
}
