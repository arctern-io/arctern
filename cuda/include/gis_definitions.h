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
class GeometryVector {
 public:
    GPUVector<char> encodeToWKB();
 private:
    GPUVector<double> data;
    GPUVector<uint32_t> offsets;
    GPUVector<Tag> tag;
    struct RuntimeHint{
        std::optional<Tag> unique_tag; 
        std::optional<int> fixed_length;
    } hints;
    size_t size;
    static GeometryVector decodeFromWKB(const std::byte* bin);
};



}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
