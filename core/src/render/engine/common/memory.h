#pragma once

#include <memory>

#include "zlibrary/memory/cuda_memory_pool.h"
#include "zlibrary/memory/main_memory_pool.h"
#include "zlibrary/memory/main_memory_pool.inl"

namespace zilliz {
namespace render {
namespace engine {


class MemManager {
 public:
    using MainMemoryPool = zilliz::lib::MainMemoryPool;
    using CUDAMemoryPool = zilliz::lib::CUDAMemoryPool;

 public:
    static MemManager &
    GetInstance() {
        static MemManager instance;
        return instance;
    }

    MainMemoryPool &
    main_memory_pool() const { return *main_memory_pool_; }

    CUDAMemoryPool &
    cuda_memory_pool() const { return *cuda_memory_pool_; }

 private:
    MemManager();

 private:
    std::shared_ptr<MainMemoryPool> main_memory_pool_;
    std::shared_ptr<CUDAMemoryPool> cuda_memory_pool_;
};


} // namespace engine
} // namespace render
} // namespace zilliz
