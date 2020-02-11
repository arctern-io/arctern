#pragma once
namespace zilliz {
namespace gis {
namespace cpp {

// must free manually
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


// must free manually
template<typename T>
T*
gpu_alloc_and_copy(const T* src, size_t size) {
    auto dst = gpu_alloc<T>(size);
    gpu_memcpy(dst, src, size);
    return dst;
}

template<typename T>
struct GpuFreeWrapper {
    void operator()(T* ptr) { gpu_free(ptr); }
};


template<typename T>
std::unique_ptr<T, GpuFreeWrapper<T>>
gpu_make_unique_array(int size) {
    return std::unique_ptr<T, GpuFreeWrapper<T>>(gpu_alloc<T>(size));
}

template<typename T>
auto
gpu_make_unique_array_and_copy(const T* src, int size)
    -> std::unique_ptr<T, GpuFreeWrapper<T>> {
    auto ptr = std::unique_ptr<T, GpuFreeWrapper<T>>(gpu_alloc<T>(size));
    gpu_memcpy(ptr.get(), src, size);
    return ptr;
}

}    // namespace cpp
}    // namespace gis
}    // namespace zilliz
