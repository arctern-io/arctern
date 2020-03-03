#pragma once
// provide mock declaration
// since nvcc can't parse arrow/api.h headers
namespace arrow {
class Array;
class Table;
}  // namespace arrow

#ifndef __CUDACC__
#include <arrow/api.h>
#include <arrow/array.h>
#endif