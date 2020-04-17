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
#include "miniz/miniz.h"
#include "stb/stb_image_write.h"

// STBIWDEF unsigned char* my_zlib_compress(unsigned char* data, int data_len, int*
// out_len,
//                                         int quality) {
//  mz_ulong buflen = mz_compressBound(data_len);
//  // Note that the returned buffer will be free'd by stbi_write_png*()
//  // with STBIW_FREE()
//  unsigned char* buf = (unsigned char*)malloc(buflen);
//  if (buf == NULL || mz_compress2(buf, &buflen, data, data_len, quality) != 0) {
//    free(buf);
//    return NULL;
//  }
//  *out_len = buflen;
//  return buf;
//}
