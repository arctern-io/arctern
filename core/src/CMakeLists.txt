#-------------------------------------------------------------------------------
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#-------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.12)

include_directories(${GIS_SOURCE_DIR})
include_directories(${GIS_ENGINE_SRC})
include_directories(${GIS_THIRDPARTY_SRC})

aux_source_directory(${GIS_ENGINE_SRC}/gis gis_files)
#aux_source_directory(${MILVUS_ENGINE_SRC}/render render_files)

set(third_party_libs
        proj
        gdal
        )


set(engine_files
        ${gis_files}
#        ${render_files}
        )

message("the eingile fiels is :${engine_files}")

set(engine_libs
        libgdal.so
        libproj.so
        )

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(engine_libs
            ${engine_libs}
            libquadmath.a
            )
endif ()

add_library(gis_engine STATIC ${engine_files})

target_link_libraries(gis_engine
        ${third_party_libs}
        ${engine_libs}
        )

