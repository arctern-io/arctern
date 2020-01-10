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

//void
//General2D::InputInit() {
//    array_vector_ = input_.array_vector;
//    vega_ = (Vega &)(input_.vega_json);
//}

void
General2D::WindowsInit(WindowParams window_params) {
//    auto window = mutable_window();
    window_ = std::make_shared<Window2D>();
    window_->set_window_params(window_params);

    window_->Init();
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

    auto bit_map = (uint8_t*)malloc(output_image_size_);
    memset(bit_map, output_image_size_, 0xff);
    auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, output_image_size_);
    auto buffer1 = std::make_shared<arrow::Buffer>(output_image_, output_image_size_);
    auto buffers = std::vector<std::shared_ptr<arrow::Buffer>>();
    buffers.emplace_back(buffer0);
    buffers.emplace_back(buffer1);

    auto data_type = arrow::uint8();
    auto array_data = arrow::ArrayData::Make(data_type, output_image_size_, buffers);
    auto array = arrow::MakeArray(array_data);

    assert(array->length() == output_image_size_);
    assert(array->type_id() == arrow::uint8()->id());

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

    return array;
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
