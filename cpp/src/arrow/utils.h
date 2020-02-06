#pragma once

#include "arrow/api.h"

#define CHECK_ARROW_STATUS(action)                        \
{                                                   \
    arrow::Status status = action;                  \
    if (!status.ok()) {                             \
        printf("%s\n", status.ToString().c_str());  \
        exit(0);                                    \
    }                                               \
}
