#pragma once
namespace zilliz {
namespace gis {
namespace cpp {

template<typename T>
T*
gpu_alloc(size_t size) {
    T* ptr;
    auto err = cudaMalloc(&ptr, size * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("error with code = " + std::to_string((int)err));
    }
    return ptr;
}

template<typename T>
void
gpu_free(T* ptr) {
    cudaFree(ptr);
}

template<typename T>
void
gpu_memcpy(T* dst, const T* src, size_t size) {
    auto err = cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        throw std::runtime_error("error with code = " + std::to_string((int)err));
    }
}


template<typename T>
T*
gpu_alloc_and_copy(const T* src, size_t size) {
    auto dst = gpu_alloc<T>(size);
    gpu_memcpy(dst, src, size);
    return dst;
}

}    // namespace gpu
}    // namespace gis
}    // namespace zilliz
