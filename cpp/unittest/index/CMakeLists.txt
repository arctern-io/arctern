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

set(index_tree_tests_src
        ${unittest_srcs}
        index_test.cpp
        )

add_executable(index_tree_tests ${index_tree_tests_src})

#add_dependencies(map_match_tests googletest_ep)

target_link_libraries(index_tree_tests
        arctern
        arrow
        gtest
        pthread
        gdal
        )

install(TARGETS index_tree_tests DESTINATION unittest)