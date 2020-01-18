#pragma once

#include "render/2d/input.h"
#include "render/window/window_egl/window_gpu_2d.h"
#include "render/window/window_osmesa/window_cpu_2d.h"

namespace zilliz {
namespace render {

class General2D {
 public:
    virtual void
    DataInit() = 0;

    virtual std::shared_ptr<uint8_t >
    Render() = 0;

    virtual void
    Shader() = 0;

    virtual void
    Draw() = 0;

    virtual void
    InputInit() = 0;

 protected:

    void
    WindowsInit(WindowParams window_params);

    void
    Finalize();

    std::shared_ptr<uint8_t >
    Output();

    void
    InitBuffer(WindowParams &window_params);

    void
    ExportImage();

 public:
    void
    set_input(Input input) { input_ = input; }

    const Input&
    input() const { return input_; }

    unsigned char*
    mutable_buffer() { return buffer_; }

    const arrow::ArrayVector&
    array_vector() const { return array_vector_; }

    int
    output_image_size() { return output_image_size_; }

 protected:
    Input input_;
    arrow::ArrayVector array_vector_;
    unsigned char *buffer_;
    unsigned char *output_image_;
    int output_image_size_;

#ifndef USE_GPU
 public:
    void
    set_window(WindowCPU2DPtr window) { window_ = window; }

    const WindowCPU2DPtr &
    window() const { return window_; }

    WindowCPU2DPtr
    mutable_window() { return window_; }
 protected:
    WindowCPU2DPtr window_;
#else
 public:
    void
    set_window(WindowGPU2DPtr window) { window_ = window; }

    const WindowGPU2DPtr &
    window() const { return window_; }

    WindowGPU2DPtr
    mutable_window() { return window_; }
 protected:
    WindowGPU2DPtr window_;
#endif

};

using General2DPtr = std::shared_ptr<General2D>;

} // namespace render
} // namespace zilliz
