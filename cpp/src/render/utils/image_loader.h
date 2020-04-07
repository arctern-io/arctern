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
#pragma once

#include <map>
#include <string>

namespace arctern {
namespace render {

class ImageLoader {
 private:
  struct ImageBuffer {
    struct ImageParams {
      int width;
      int height;
    };

    ImageParams image_params;
    unsigned char* buffer;
  };

 public:
  static ImageLoader& GetInstance() {
    static ImageLoader instance;
    return instance;
  }

  ImageBuffer Load(const std::string& file_path);

  void LoadDir(const std::string& file_path);

  const std::map<std::string, ImageBuffer>& image_buffers() const {
    return image_buffers_;
  }

 private:
  ImageLoader() = default;

 private:
  std::map<std::string, ImageBuffer> image_buffers_;
};

}  // namespace render
}  // namespace arctern
