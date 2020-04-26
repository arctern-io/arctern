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

set(render_tests_src
        ${unittest_srcs}
		pointmap_test.cpp
	    heatmap_test.cpp
		choropleth_map_test.cpp
		coordinate_projection_test.cpp
		weighted_pointmap_test.cpp
		iconviz_test.cpp
		)

add_executable(render_tests ${render_tests_src})

target_link_libraries(render_tests
        arctern
        arrow
        gtest
        pthread
        )

install(TARGETS render_tests DESTINATION unittest)

