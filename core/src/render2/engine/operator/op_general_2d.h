#pragma once

#include "operator.h"

#include <sys/time.h>


namespace zilliz {
namespace render {
namespace engine {


class OpGeneral2D : public Operator {
 public:
    OpGeneral2D();
    virtual ~OpGeneral2D();

 public:
    virtual DatasetPtr
    Render() = 0;

 protected:
    void
    Init();

    void
    Finalize();

    DatasetPtr
    Output();

 private:
    void
    InitBuffer(WindowParams &window_params);

    void
    ExportImage();

    unsigned char *
    mutable_buffer() { return buffer_; }

 protected:
    struct timeval tstart, tend;
    unsigned char *buffer_;
    unsigned char *output_image_;
    int output_image_size_;
};


} // namespace engine
} // namespace render
} // namespace zilliz
