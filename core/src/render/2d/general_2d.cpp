#include <iostream>
#include "arrow/buffer.h"
#include "arrow/type.h"
#include "arrow/vendored/string_view.hpp"
#include "render/2d/general_2d.h"

#define STBIW_ZLIB_COMPRESS my_zlib_compress
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb/stb_image_write.h"



namespace zilliz {
namespace render {

void
General2D::WindowsInit(WindowParams window_params) {
    auto window = mutable_window();
    window = std::make_shared<Window2D>();
    window->set_window_params(window_params);

    window->Init();
    InitBuffer(window_params);
}

void
General2D::Finalize() {
    eglSwapBuffers(mutable_window()->mutable_egl_dpy(),
                   mutable_window()->mutable_egl_surf());
    auto width = mutable_window()->window_params().width();
    auto height = mutable_window()->window_params().height();

    for (int i = 0; i < width * height * 4; i++) {
        mutable_buffer()[i] = 0;
    }

    glReadPixels(0,
                 0,
                 width,
                 height,
                 GL_RGBA,
                 GL_UNSIGNED_INT_8_8_8_8_REV,
                 mutable_buffer());
    window()->Terminate();
}

std::shared_ptr<arrow::Array>
General2D::Output() {
    // export image to memory
    ExportImage();

    auto data_type = std::make_shared<arrow::DataType>(arrow::UInt8Type());
    auto buffers = std::vector<std::shared_ptr<arrow::Buffer>>();
    buffers.push_back(std::make_shared<arrow::Buffer>(arrow::Buffer(output_image_, output_image_size_)));
    auto array_data = arrow::ArrayData(data_type, output_image_size_, buffers);
    auto output = std::make_shared<arrow::Array>();
    output->data() = std::make_shared<arrow::ArrayData>(array_data);

    auto write_image = true;
    if (write_image) {
        std::cout << "******************" << output_image_size_ << "******************" << std::endl;
        FILE *f = fopen("/tmp/offscreen.png", "wb");
        if (!f) {
            std::cout << "export png error";
        } else {
            fwrite(output_image_, 1, output_image_size_, f);
            fclose(f);
        }
    }

    return output;
}

void
General2D::InitBuffer(zilliz::render::WindowParams &window_params) {
    buffer_ = (unsigned char *) calloc(size_t(window_params.width() * window_params.height()), 4);

}

void
General2D::ExportImage() {
    auto &window_params = window_->window_params();

    auto pixels = buffer_ + (int) (window_params.width() * 4 * (window_params.height() - 1));
    auto stride_bytes = -(window_params.width() * 4);

    output_image_ = stbi_write_png_to_mem(pixels,
                                          stride_bytes,
                                          window_params.width(),
                                          window_params.height(),
                                          4,
                                          &output_image_size_);

}

} //namespace render
} //namespace zilliz
