#include <iostream>

#include "arrow/buffer.h"
#include "arrow/type.h"
#include "arrow/vendored/string_view.hpp"
#include "render/2d/general_2d.h"
#include "render/utils/my_zlib_compress.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb/stb_image_write.h"


namespace zilliz {
namespace render {


void
General2D::WindowsInit(WindowParams window_params) {
#ifndef USE_GPU
    window_ = std::make_shared<WindowCPU2D>();
    window_->set_window_params(window_params);
    // We'v been init buffer in Window.
    window_->Init();
#else
    window_ = std::make_shared<WindowGPU2D>();
    window_->set_window_params(window_params);
    window_->Init();
    InitBuffer(window_params);
#endif
}

void
General2D::Finalize() {
#ifndef USE_GPU
    // OSMesa bind image buffer to OSMesaContext,
    // buffer have been written after glFinish(),
    // so we don't need SwapBuffer or ReadPixels here.
    buffer_ = window_->mutable_buffer();
    window_->Terminate();
#else
    eglSwapBuffers(mutable_window()->mutable_egl_dpy(),
                   mutable_window()->mutable_egl_surf());
    auto width = window()->window_params().width();
    auto height = window()->window_params().height();

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
#endif
}

std::shared_ptr<uint8_t>
General2D::Output() {
    // export image to memory
    ExportImage();

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

    return std::shared_ptr<uint8_t>(output_image_);
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
