#pragma once

#include <iostream>
namespace zilliz {
namespace lib {

#define CUDA_DOMAIN "CUDA"
using ErrorCode = int32_t;

class Exception : public std::exception {
 public:
    Exception(ErrorCode error_code,
              const std::string &domain,
              const std::string &message = std::string())
        : error_code_(error_code), domain_(domain), message_(message) {}

 public:
    ErrorCode error_code() const { return error_code_; }

    std::string domain() const { return domain_; }

    virtual const char *what() const noexcept { return message_.c_str(); }

 private:
    ErrorCode error_code_;
    std::string domain_;
    std::string message_;
};

class CudaException : public Exception {
 public:
    CudaException(ErrorCode error_code, const std::string &message = nullptr)
        : Exception(error_code, CUDA_DOMAIN, message) {}
};

#define CHECK_CUDA(function)                                                         \
    do {                                                                             \
        ErrorCode result = static_cast<ErrorCode>(function);                         \
        if (result != CUDA_SUCCESS) {                                                \
            std::string msg = "cuda internal error";                                 \
            throw CudaException(result, msg);                                        \
        }                                                                            \
    } while (false);
}
}