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

with open('/tmp/z_curve.json', 'w') as file:
    # y = 150
    for i in range(100, 200):
        file.write('{"x": %d, "y": %d}\n' % (i, 150))

    # y = x - 50
    for i in range(100, 200):
        file.write('{"x": %d, "y": %d}\n' % (i, i - 50))

    # y = 50
    for i in range(100, 200):
        file.write('{"x": %d, "y": %d}\n' % (i, 50))
