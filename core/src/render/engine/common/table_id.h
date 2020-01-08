#pragma once

#include "render/dependences/common/table_id.h"
#include "render/dependences/common/table_id_attr_ex.h"


namespace zilliz {
namespace render {
namespace engine {

using ChunkID = zilliz::common::TableIDAttrEx;
using ColumnID = zilliz::common::TableIDAttrEx;
using FragmentID = zilliz::common::TableIDAttrEx;
using TableID = zilliz::common::TableIDAttrEx;

using ColumnField = zilliz::common::TableID::ColumnField;
using FragmentField = zilliz::common::TableID::FragmentField;


inline zilliz::common::TableIDAttrEx
GetTableIDAttrEx(zilliz::common::TableID id,
                 zilliz::common::TableIDAttrEx::AttrType attr_type) {

    zilliz::common::TableIDAttrEx attr_ex_id(id);
    attr_ex_id.set_type(zilliz::common::IDType::kTableAttrEx);
    attr_ex_id.set_attr_type(attr_type);
    return attr_ex_id;
}


} // namespace engine
} // namespace render
} // namespace zilliz
