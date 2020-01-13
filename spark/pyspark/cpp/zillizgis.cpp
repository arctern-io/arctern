#define PY_SSZIE_T_CLEAN
#include "Python.h"
#include "arrow/python/pyarrow.h"
#include "arrow/api.h"
#include <stdio.h>
#include <stdint.h>
#include <string>

#ifdef _GLIBCXX_USE_CXX11_ABI
#undef _GLIBCXX_USE_CXX11_ABI
#endif

#define _GLIBCXX_USE_CXX11_ABI 0

#define CHECK_STATUS(action)                        \
{                                                   \
    arrow::Status status = action;                  \
    if (!status.ok()) {                             \
        printf("%s\n", status.ToString().c_str());  \
        exit(0);                                    \
    }                                               \
}

extern std::shared_ptr<arrow::Array>
make_point(std::shared_ptr<arrow::Array>, std::shared_ptr<arrow::Array>);

static PyObject* ST_POINT(PyObject* self, PyObject* obj) {
    arrow::py::import_pyarrow();

    PyObject *parsed_x, *parsed_y;
    if (!PyArg_ParseTuple(obj, "OO", &parsed_x, &parsed_y)) { // "O" in upper-case
        printf("parse error!\n");
        return NULL;
    }

    std::shared_ptr<arrow::Array> arr_x, arr_y;
    CHECK_STATUS(arrow::py::unwrap_array(parsed_x, &arr_x));
    CHECK_STATUS(arrow::py::unwrap_array(parsed_y, &arr_y));

    return arrow::py::wrap_array(make_point(arr_x, arr_y));
}

static PyMethodDef ZillizGisExt[] = {
    {"ST_POINT", ST_POINT, METH_VARARGS, "ST_POINT"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ZillizGisModule = {
    PyModuleDef_HEAD_INIT,
    "zillizgis",
    NULL,
    -1,
    ZillizGisExt
};

PyMODINIT_FUNC
PyInit_zillizgis(void) {
    arrow::py::import_pyarrow();
    return PyModule_Create(&ZillizGisModule);
}
