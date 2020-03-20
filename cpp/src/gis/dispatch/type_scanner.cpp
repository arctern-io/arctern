
#include "gis/dispatch/type_scanner.h"

namespace arctern {
namespace gis {
namespace dispatch {

void MaskResult::AppendRequire(const GeometryTypeScanner& scanner,
                               const GroupedWkbTypes& supported) {
  auto type_masks = scanner.Scan();
  auto status =
      !type_masks->is_unique_type
          ? Status::kMixed
          : type_masks->unique_type == supported ? Status::kOnlyTrue : Status::kOnlyFalse;
  if ((int)this->status_ < (int)status) {
    return;
  } else if ((int)this->status_ > (int)status) {
    this->status_ = status;
    if (status == Status::kMixed) {
      this->mask_ = std::move(*type_masks).get_mask(supported);
      return;
    } else {
      this->mask_.clear();
      this->mask_.shrink_to_fit();
      return;
    }
  } else {
    if (status != Status::kMixed) {
      return;
    }
    auto& mask = type_masks->get_mask(supported);
    assert(mask.size() == this->mask_.size());
    bool has_true = false;
    for (auto i = 0; i < mask.size(); ++i) {
      bool flag = this->mask_[i] && mask[i];
      this->mask_[i] = flag;
      has_true = has_true || flag;
    }

    // downgrade to last
    if (!has_true) {
      this->status_ = Status::kOnlyFalse;
      this->mask_.clear();
      this->mask_.shrink_to_fit();
    }
  }
}

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
