#pragma once

namespace zilliz {
namespace lib {
using EnumType = uint64_t;
enum class ValueType : EnumType {
    kValUnknown = 0,

    kFixedLengthTypeBegin = 100,
    kValVoid = kFixedLengthTypeBegin,
    kValPointer,
    kValBool = 200,
    kValBool8,
    kValInt8 = 300,
    kValInt16,
    kValInt32,
    kValInt64,
    kValUInt8 = 400,
    kValUInt16,
    kValUInt32,
    kValUInt64,
    kValFloat = 500,
    kValDouble,
    kValNumeric,
    kValTime = 600,
    kValTimestamp,
    kValDate,
    kValInterval,
    kValPoint = 700,
    kValArrayInt16 = 800,
    kValArrayInt32,
    kValArrayInt64,
    kValArrayFloat,
    kValArrayDouble,
    kValArrayNumeric,
    kFixedLengthTypeEnd,

    kVarLengthTypeBegin = 900,
    kValText = kVarLengthTypeBegin,    // unlimited-length string
    kValChar,                          // single character
    kValVarchar,                       // variable-length string with length limit
    kValBpChar,                        // fixed-length string with blank padded
    kValName,                          // aka PG: NAMEOID 19
    kValBinary,
    kVarLengthTypeEnd

};    // ValueType
} //lib
} //zilliz