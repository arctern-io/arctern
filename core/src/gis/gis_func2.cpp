#include "arrow/api.h"


using namespace std;

namespace gis {
shared_ptr<arrow::Array> gis_func2(shared_ptr<arrow::Array> arr_ptr1, shared_ptr<arrow::Array> arr_ptr2) {
    auto int_array1 = static_pointer_cast<arrow::Int64Array>(arr_ptr1);
    auto int_array2 = static_pointer_cast<arrow::Int64Array>(arr_ptr2);

    auto length = int_array1->length();
    assert(length == int_array2->length());

    arrow::StringBuilder builder;
    arrow::Status status;
    shared_ptr<arrow::Array> out_array;

    for (int i = 0; i < 1000000; ++i) {
        status = builder.Append(to_string(int_array1->Value(0)) + to_string(int_array2->Value(1)));
    }

    status = builder.Finish(&out_array);

    return out_array;
}
}