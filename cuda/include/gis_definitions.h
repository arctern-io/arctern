#include <vector>
#include <set>
#include <optional>
#include <array>
#include <tuple>
#include <cassert>
using std::vector;
template<typename T>
using GPUVector = vector<int>;    // TODO: use gpu vector, now just placeholder
enum class Tag {
    Invalid = 0,
    Point1D = 1
};    // TODO: use enum of GIS, now just placeholder


namespace zilliz {
namespace gis {
namespace cpp {
class GeometryVectorBase {
 public:
    virtual ~GeometryVectorBase() {}

    // size of individual geometries
    virtual uint64_t size() = 0;

    // test if tag are varible
    virtual std::optional<Tag> unique_tag() = 0;

    // test if layout length is fixed
    virtual bool is_fixed_layout() = 0;
};


inline constexpr Tag
get_tag_for_points(int dimension) {
    switch (dimension) {
        case 1:
            return Tag::Point1D;
        default:
            break;
    }
}

template<int dimension_>
class PointVector : GeometryVectorBase {
 public:
    constexpr int dimension = dimension_;
    constexpr Tag tag = get_tag_for_points(dimension);
    virtual uint64_t size() { return data[0].size(); }

 private:
    std::array<GPUVector<double>, dimension> data;
};

using Point2DVector = PointVector<2>;

template<int dimension_>
class LineStringVector:


}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
