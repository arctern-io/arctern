#pragma once

#include "zcommon/error/error.h"
#include "render/engine/common/log.h"
#include "zlibrary/error/error.h"


#define RENDER_ENGINE_DOMAIN "RENDER ENGINE"

namespace zilliz {
namespace render {
namespace engine {


using ErrorCode = zilliz::common::ErrorCode;

using zilliz::lib::CUDA_SUCCESS;
using zilliz::lib::CudaException;

constexpr ErrorCode ERROR_CODE_BASE = 0x45000;
constexpr ErrorCode ERROR_CODE_END = 0x46000;


constexpr ErrorCode
ToGlobalErrorCode(const ErrorCode error_code) {
    return zilliz::lib::ToGlobalErrorCode(error_code, ERROR_CODE_BASE);
}

constexpr ErrorCode UNKNOWN_PLAN_TYPE = ToGlobalErrorCode(0x001);
constexpr ErrorCode ILLEGAL_VEGA_FORMAT = ToGlobalErrorCode(0x002);
constexpr ErrorCode UNSUPPORTED_IMAGE_FORMAT = ToGlobalErrorCode(0x003);
constexpr ErrorCode INPUT_NOT_FOUND = ToGlobalErrorCode(0x004);
constexpr ErrorCode CREATE_RESPONSE_DATA_NULL = ToGlobalErrorCode(0x005);
constexpr ErrorCode LABEL_NOT_FOUND = ToGlobalErrorCode(0x006);
constexpr ErrorCode COLOR_STYLE_NOT_FOUND = ToGlobalErrorCode(0x007);
constexpr ErrorCode VALUE_TYPE_NOT_FOUND = ToGlobalErrorCode(0x008);
constexpr ErrorCode UNKNOWN_STRING_TYPE = ToGlobalErrorCode(0x009);
constexpr ErrorCode ILLEGAL_COLUMN_TYPE = ToGlobalErrorCode(0x00a);
constexpr ErrorCode ILLEGAL_WINDOW_SIZE = ToGlobalErrorCode(0x00b);
constexpr ErrorCode NULL_PTR = ToGlobalErrorCode(0x00c);
constexpr ErrorCode CAST_FAILED = ToGlobalErrorCode(0x00d);
constexpr ErrorCode FILE_PATH_NOT_FOUND = ToGlobalErrorCode(0x00e);
constexpr ErrorCode UNINITIALIZED = ToGlobalErrorCode(0x00f);


class RenderEngineException : public zilliz::common::Exception {
 public:
    explicit
    RenderEngineException(ErrorCode error_code,
                          const std::string &message = nullptr)
        : Exception(error_code, RENDER_ENGINE_DOMAIN, message) {}
};


} // namespace engine
} // namespace render
} // namespace zilliz

#define THROW_RENDER_ENGINE_ERROR(err_code, err_msg)                  \
    do {                                                       \
        RENDER_ENGINE_LOG_ERROR << err_msg;                           \
        throw RenderEngineException(err_code, err_msg);              \
    } while(false);
