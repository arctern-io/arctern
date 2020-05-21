/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <dirent.h>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "render/utils/image/image_loader.h"

namespace arctern {
namespace render {

ImageLoader::ImageBuffer ImageLoader::Load(const std::string& file_path) {
  int width, height, channels_in_file;

  stbi_set_flip_vertically_on_load(true);

  auto pixel =
      stbi_load(file_path.c_str(), &width, &height, &channels_in_file, STBI_rgb_alpha);

  ImageBuffer image_buffer{};
  image_buffer.buffer = pixel;
  image_buffer.image_params.width = width;
  image_buffer.image_params.height = height;

  return image_buffer;
}

void ImageLoader::LoadDir(const std::string& file_path) {
  image_buffers_.clear();

  int width, height, channels_in_file;

  DIR* dir;
  struct dirent* ent;

  if ((dir = opendir(file_path.c_str())) == nullptr) {
    std::string err_msg = "Cannot find images file path: " + file_path;
    throw std::runtime_error(err_msg);
  }

  stbi_set_flip_vertically_on_load(true);

  while ((ent = readdir(dir)) != nullptr) {
    if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) {
      continue;
    }

    std::string file_name = file_path + ent->d_name;

    auto pixel =
        stbi_load(file_name.c_str(), &width, &height, &channels_in_file, STBI_rgb_alpha);

    ImageBuffer image_buffer{};
    image_buffer.buffer = pixel;
    image_buffer.image_params.width = width;
    image_buffer.image_params.height = height;

    image_buffers_.emplace(ent->d_name, image_buffer);
  }

  closedir(dir);
}

}  // namespace render
}  // namespace arctern
