# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2

__all__ = [
    "save_png",
    "diff_png"
]

def save_png(hex_str, file_name):
    import base64
    binary_string = base64.b64decode(hex_str)
    with open(file_name, 'wb') as png:
        png.write(binary_string)

def diff_png(baseline_png, compared_png, precision=0.00005):
    baseline_info = cv2.imread(baseline_png, cv2.IMREAD_UNCHANGED)
    compared_info = cv2.imread(compared_png, cv2.IMREAD_UNCHANGED)
    baseline_y, baseline_x = baseline_info.shape[0], baseline_info.shape[1]
    baseline_size = baseline_info.size

    compared_y, compared_x = compared_info.shape[0], compared_info.shape[1]
    compared_size = compared_info.size
    if compared_y != baseline_y or compared_x != baseline_x or compared_size != baseline_size:
        return False

    diff_point_num = 0
    for i in range(baseline_y):
        for j in range(baseline_x):
            baseline_rgba = baseline_info[i][j]
            compared_rgba = compared_info[i][j]

            baseline_rgba_len = len(baseline_rgba)
            compared_rgba_len = len(compared_rgba)
            if baseline_rgba_len != compared_rgba_len or baseline_rgba_len != 4:
                return False
            if compared_rgba[3] == baseline_rgba[3] and baseline_rgba[3] == 0:
                continue

            is_point_equal = True
            for k in range(3):
                tmp_diff = abs((int)(compared_rgba[k]) - (int)(baseline_rgba[k]))
                if tmp_diff > 1:
                    is_point_equal = False

            if is_point_equal == False:
                diff_point_num += 1

    if ((float)(diff_point_num) / (float)(baseline_size)) <= precision:
        return True
    else:
        return False