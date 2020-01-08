#include <sys/time.h>
#include "render/engine/window/window2d.h"
#include "render/engine/operator/operator.h"
#include "render/engine/operator/op_general_2d.h"
#include "render/engine/common/log.h"
#include "render/engine/common/error.h"

#include "zcommon/config/megawise_config.h"


#define STBIW_ZLIB_COMPRESS my_zlib_compress
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../../../../thirdparty/stb/stb_image_write.h"


namespace zilliz {
namespace render {
namespace engine {


OpGeneral2D::OpGeneral2D()
    : Operator(), buffer_(nullptr), output_image_(nullptr), output_image_size_(0) {
}

OpGeneral2D::~OpGeneral2D() {
    if (buffer_ != nullptr) {
        free(buffer_);
    }
    if (output_image_ != nullptr) {
        free(output_image_);
    }
}

void
OpGeneral2D::Init() {

    gettimeofday(&tstart, nullptr);

    auto window_params = plan()->window_params();
    auto plan_type = plan()->plan_type();

    auto &window = mutable_window();
    window = std::make_shared<Window2D>();
    window->set_window_params(window_params);

    if (plan_type == RenderPlan::Type::k2D) {
        window->set_window_type(WindowType::k2D);
    } else {
//        THROW_RENDER_ENGINE_ERROR(UNKNOWN_PLAN_TYPE, "unknown render plan type.")
    }

    window->Init();
    InitBuffer(window_params);
}


void
OpGeneral2D::Finalize() {
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


DatasetPtr
OpGeneral2D::Output() {

    auto output = std::make_shared<Dataset>();
    auto &plan_node = plan()->root_plan_node();

    TableID output_id(plan_node->output_id().db_id(),
                      plan_node->output_id().table_id());
    output_id.set_attr_type(TableID::AttrType::kMeta);

    // set table info
    // we have only one output image, such that num_fragment = 1, num_column = 1
    auto &table_info = output->table_info_[output_id];
    table_info.fragments.push_back(0);
    table_info.columns.push_back(0);

    output_id.set_column_id(common::TableID::kInvalidColumnField);
    output_id.set_fragment_id(0);
    // set fragment info
    // we have only one output image, such that num_rows = 1.
    zilliz::bulletin::FragmentBoardPtr
        fragment_board = std::make_shared<zilliz::bulletin::FragmentBoard>(output_id);
    fragment_board->set_num_rows(1);
    output->meta_map_[output_id] = std::static_pointer_cast<Dataset::Meta>(fragment_board);

    // export image to memory
    ExportImage();

    output_id.set_column_id(0);
    output_id.set_attr_type(TableID::AttrType::kData);
    // copy image to dataset
    auto create_resp = data_client()->Create(output_id,
                                             chewie::MM_Data_Partition_Id,
                                             output_image_size_,
                                             chewie::CacheHint::fNotDroppable);
    if (create_resp->buffer->data() == nullptr) {
//        THROW_RENDER_ENGINE_ERROR(CREATE_RESPONSE_DATA_NULL, "chewie buffer data created is null.")
    }
    std::memcpy((unsigned char *) (create_resp->buffer->data()), output_image_, size_t(output_image_size_));

    auto write_image = common::megawise::DevCfg::render_engine::write_image();
    if (write_image) {
        std::cout << "******************" << output_image_size_ << "******************" << std::endl;
        FILE *f = fopen("/tmp/offscreen.png", "wb");
        if (!f) {
//            RENDER_ENGINE_LOG_ERROR << "export png error";
        } else {
            fwrite(output_image_, 1, output_image_size_, f);
            fclose(f);
        }
    }

    data_client()->Seal(output_id, chewie::MM_Data_Partition_Id);
    data_client()->Release(output_id, chewie::MM_Data_Partition_Id);

    // TODO:: add print_run_time bool in zcommon
    auto print_run_time = common::megawise::DevCfg::render_engine::write_image();
    if (print_run_time) {
        gettimeofday(&tend, NULL);
        auto timer = 1000000 * (tend.tv_sec - tstart.tv_sec) + tend.tv_usec - tstart.tv_usec;
        std::cout << "render engine time: " << timer/1000  << "ms"<< std::endl;
    }

    return output;
}


void
OpGeneral2D::InitBuffer(WindowParams &window_params) {
    buffer_ = (unsigned char *) calloc(size_t(window_params.width() * window_params.height()), 4);
}

void
OpGeneral2D::ExportImage() {

    auto &window_params = plan()->window_params();

    if (plan()->image_format()->type == ImageFormat::Type::kPNG) {

        auto pixels = buffer_ + (int) (window_params.width() * 4 * (window_params.height() - 1));
        auto stride_bytes = -(window_params.width() * 4);

        output_image_ = stbi_write_png_to_mem(pixels,
                                              stride_bytes,
                                              window_params.width(),
                                              window_params.height(),
                                              4,
                                              &output_image_size_);
    }
}


} // namespace engine
} // namespace render
} // namespace zilliz