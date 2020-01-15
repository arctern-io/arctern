#include "render_builder.h"

namespace zilliz {
namespace render {

std::pair<std::shared_ptr<uint8_t >,int64_t>
pointmap(std::shared_ptr<uint32_t > arr_x, std::shared_ptr<uint32_t > arr_y, int64_t num) {
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

    return std::make_pair(point_map.Render(), point_map.num_vertices());
}

} //namespace render
} //namespace zilliz