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

set(gis_cuda_test_srcs
        transforms_tests.cpp
        functor_distance_tests.cpp
        functor_area_tests.cpp
        functor_point_tests.cpp
        functor_within_tests.cpp
        functor_length_tests.cpp
        functor_envelope_tests.cpp
        functor_relate_tests.cpp
        arrow_conversion_tests.cpp
        host_relation_tests.cpp
        ${unittest_srcs}
        )
add_executable(gis_cuda_tests ${gis_cuda_test_srcs})

target_link_libraries(gis_cuda_tests arctern gtest)

install(TARGETS gis_cuda_tests DESTINATION unittest)
