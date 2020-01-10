#pragma once

#include "render/2d/input.h"
#include "render/window/window2d.h"

namespace zilliz {
namespace render {

class General2D {
 public:
//    General2D();

    virtual void
    DataInit() = 0;

    virtual std::shared_ptr<arrow::Array>
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

    std::shared_ptr<arrow::Array>
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

    void
    set_window(Window2DPtr window) { window_ = window; }

    const Window2DPtr &
    window() const { return window_; }

    Window2DPtr
    mutable_window() { return window_; }

    unsigned char*
    mutable_buffer() { return buffer_; }

    const arrow::ArrayVector&
    array_vector() const { return array_vector_; }

//    const Vega&
//    vega() const { return vega_; }

 private:
    Input input_;
    arrow::ArrayVector array_vector_;
//    Vega vega_;
    Window2DPtr window_;
    unsigned char *buffer_;
    unsigned char *output_image_;
    int output_image_size_;

};

using General2DPtr = std::shared_ptr<General2D>;

} // namespace render
} // namespace zilliz
