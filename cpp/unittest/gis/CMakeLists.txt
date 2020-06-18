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

set(gis_tests_src
        ${unittest_srcs}
        gis_functions_tests.cpp
        parser_tests.cpp)

add_executable(gis_tests ${gis_tests_src})

#add_dependencies(gis_tests googletest_ep)

target_link_libraries(gis_tests
        arctern
        arrow
        gtest
        pthread
        gdal
        )

install(TARGETS gis_tests DESTINATION unittest)

add_subdirectory(gdal)
add_subdirectory(spatial_join)
if(BUILD_WITH_GPU)
    add_subdirectory(cuda)
endif()
