#pragma GCC diagnostic ignored "-Wunused-function"

#include <dirent.h>

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_IMPLEMENTATION

#include "stb/stb_image.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb/stb_image_write.h"

#include "render/engine/common/error.h"
#include "render/engine/common/log.h"
#include "render/engine/image/loader/image_loader.h"


namespace zilliz {
namespace render {
namespace engine {


void ImageLoader::Load(const std::string &file_path) {

    image_buffers_.clear();

    int width, height, channels_in_file;

    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(file_path.c_str())) == NULL) {
        RENDER_ENGINE_LOG_WARNING << "Cannot find images file path: " << file_path;
        return;
    }

    while ((ent = readdir(dir)) != NULL) {

        if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) {
            continue;
        }

        std::string file_name = file_path + ent->d_name;

        auto pixel = stbi_load(file_name.c_str(),
                                           &width,
                                           &height,
                                           &channels_in_file,
                                           STBI_rgb_alpha);

        ImageBuffer image_buffer;
        image_buffer.buffer = pixel;
        image_buffer.image_params.width = width;
        image_buffer.image_params.height = height;

        image_buffers_.emplace(ent->d_name, image_buffer);
    }

    closedir(dir);
}


} // namespace engine
} // namespace render
} // namespace zilliz