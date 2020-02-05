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

    // test if layout is fixed
    virtual bool is_fixed_layout() = 0;

    // return possible tags
    virtual std::set<Tag> all_tags() = 0;
};

class UniqueTagGeometryVector : public GeometryVectorBase {
 public:
    UniqueTagGeometryVector(Tag tag) : tag_(tag) {}
    std::set<Tag> all_tags() override { return {tag_}; }
    Tag tag() { return tag_; }

 private:
    const Tag tag_;
};

inline constexpr std::optional<int>
get_tag_length(Tag tag) {
    // TODO: placeholder, return std::nullopt for tag of variable length;
    // now only Point1D
    return sizeof(double);
}

class FixedLayoutGeometryVector : public UniqueTagGeometryVector {
 public:
    FixedLayoutGeometryVector(Tag tag) : UniqueTagGeometryVector(tag) {
        // check if fixed
        assert(get_tag_length(tag).has_value());
    }
};

class VariableLayoutUniqueTagGeometryVector : public UniqueTagGeometryVector {
 public:
    VariableLayoutUniqueTagGeometryVector(Tag tag): UniqueTagGeometryVector(tag) {
        // check if fixed
        assert(!get_tag_length(tag).has_value());
    }
};

class VariableTagGeometryVector: public GeometryVectorBase {
       
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

template<int dimension>
class Points : FixedLayoutGeometry {
 public:
    std::array<GPUVector<double>, demension> point_arrs;
};


}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
