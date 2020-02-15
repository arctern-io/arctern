#pragma once
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <memory>
#include <cuda.h>

#include <cuda_runtime.h>
namespace zilliz {
namespace gis {
namespace cuda {

// must free manually
template<typename T>
T*
GpuAlloc(size_t size) {
    T* ptr;
    auto err = cudaMalloc(&ptr, size * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error("error with code = " + std::to_string((int)err) + "<<" +
                                 cudaGetErrorString(err));
    }
    return ptr;
}

template<typename T>
void
GpuFree(T* ptr) {
    cudaFree(ptr);
}

template<typename T>
void
GpuMemcpy(T* dst, const T* src, int size) {
    if (size == 0) {
        return;
    }
    assert(size > 0);
    auto err = cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        throw std::runtime_error("error with code = " + std::to_string((int)err) + "<<" +
                                 cudaGetErrorString(err));
    }
}


// must free manually
template<typename T>
T*
GpuAllocAndCopy(const T* src, int size) {
    auto dst = GpuAlloc<T>(size);
    GpuMemcpy(dst, src, size);
    return dst;
}

template<typename T>
struct GpuFreeWrapper {
    void operator()(T* ptr) { GpuFree(ptr); }
};


template<typename T>
std::unique_ptr<T, GpuFreeWrapper<T>>
GpuMakeUniqueArray(int size) {
    return std::unique_ptr<T, GpuFreeWrapper<T>>(GpuAlloc<T>(size));
}

template<typename T>
auto
GpuMakeUniqueArrayAndCopy(const T* src, int size)
    -> std::unique_ptr<T, GpuFreeWrapper<T>> {
    auto ptr = std::unique_ptr<T, GpuFreeWrapper<T>>(GpuAlloc<T>(size));
    GpuMemcpy(ptr.get(), src, size);
    return ptr;
}

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
