#include "render_builder.h"

namespace zilliz {
namespace render {
std::shared_ptr<arrow::Array>
get_pointmap(arrow::ArrayVector input_array) {
   PointMap point_map;
   Input input;
   input.array_vector = input_array;
   input.vega = "{\n"
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
    point_map.set_input(input);
    return point_map.Render();
}

} //namespace render
} //namespace zilliz