
#include "gis/dispatch/type_scanner.h"

namespace arctern {
namespace gis {
namespace dispatch {

void MaskResult::AppendRequire(const GeometryTypeMasks& type_masks,
                               const GroupedWkbTypes& supported) {
  auto status = [&]() {
//    if (type_masks.is_unique_type) {
      if (type_masks.unique_type == supported) {
        return Status::kOnlyTrue;
      } else {
        return Status::kOnlyFalse;
      }
    } else {
      return Status::kMixed;
    }
  }();
}

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
