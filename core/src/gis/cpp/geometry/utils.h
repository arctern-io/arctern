#pragma once

namespace zilliz {
namespace gis {
namespace cpp {
namespace gemetry {

#define CHECK_STATUS(action)                        \
{                                                   \
    arrow::Status status = action;                  \
    if (!status.ok()) {                             \
        printf("%s\n", status.ToString().c_str());  \
        exit(0);                                    \
    }                                               \
}

}
}
}
}
