#pragma once


namespace zilliz {
namespace render {


/***
 * TODO: add comments
 */
struct Vega {
 public:
    Vega(const std::string& json);

    std::string ToString();

 private:
    // vega json to vega struct
    Parse(const std::string& json);
};


} // namespace render
} // namespace zilliz
