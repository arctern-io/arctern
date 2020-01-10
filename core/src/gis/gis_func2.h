
#ifndef GIS_GIS_FUNC2_H
#define GIS_GIS_FUNC2_H

#include "arrow/api.h"


using namespace std;

namespace gis {

shared_ptr<arrow::Array>
gis_func2(shared_ptr<arrow::Array> arr_ptr1, shared_ptr<arrow::Array> arr_ptr2);

}
#endif //GIS_GIS_FUNC2_H
