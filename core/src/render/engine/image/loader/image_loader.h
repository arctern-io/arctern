#pragma once

#include <vector>
#include <string>
#include <map>


namespace zilliz {
namespace render {
namespace engine {


class ImageLoader {
 private:
    struct ImageBuffer {
        struct ImageParams {
            float width;
            float height;
        };

        ImageParams image_params;
        unsigned char *buffer;
    };

 public:
    static ImageLoader &
    GetInstance() {
        static ImageLoader instance;
        return instance;
    }

    void Load(const std::string &file_path);

    const std::map<std::string, ImageBuffer> &
    image_buffers() const { return image_buffers_; }

 private:
    ImageLoader() = default;

 private:
    std::map<std::string, ImageBuffer> image_buffers_;
};


} // namespace engine
} // namespace render
} // namespace zilliz