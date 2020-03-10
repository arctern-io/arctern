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

include_directories(${ARCTERN_SOURCE_DIR})
include_directories(${ARCTERN_ENGINE_SRC})
include_directories(${ARCTERN_THIRDPARTY_SRC})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(render_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/scatter_plot/pointmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/choropleth_map/choropleth_map.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/general_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/heatmap/heatmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/heatmap/set_color.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega_scatter_plot/vega_circle2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega_choropleth_map/vega_choropleth_map.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega_heatmap/vega_heatmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/color/color.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/color/color_gradient.cpp
	)

set(cpu_render_src
	${CMAKE_CURRENT_SOURCE_DIR}/render/window/window_osmesa/window_cpu_2d.cpp
	)

set(gpu_render_src
	${CMAKE_CURRENT_SOURCE_DIR}/render/window/window_egl/window_gpu_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/cuda/heatmap.cu
	)

set(gis_src
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gis_functions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gdal/gis_functions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gdal/geometry_visitor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gdal/type_scan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gdal/format_conversion.cpp
        )

set(cuda_gis_src
	${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/wkb/wkb_transforms.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/conversion/arrow_decode.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/conversion/arrow_decode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/conversion/arrow_encode.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/conversion/arrow_encode.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/common/gis_definitions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/common/gis_definitions.cu
	${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/functor/st_distance.cu
	${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/functor/st_within.cu
	${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/functor/st_length.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/functor/st_area.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/functor/st_envelope.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/functor/st_point.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/cuda/gis_functions.cpp
        )

set(arrow_src
        ${CMAKE_CURRENT_SOURCE_DIR}/arrow/render_api.cpp
        )

set(LIBRARY_TYPE SHARED)
if (STATIC_LIB_SWITCH)
    set(LIBRARY_TYPE STATIC)
endif (STATIC_LIB_SWITCH)

if (BUILD_WITH_GPU)
    add_library(
        arctern ${LIBRARY_TYPE}
        ${gis_src}
        ${cuda_gis_src}
        ${render_src}
        ${gpu_render_src}
	${arrow_src}
	)
else ()
    add_library(
        arctern ${LIBRARY_TYPE}
	${gis_src}
	${render_src}
	${cpu_render_src}
	${arrow_src}
	)
endif ()

set(third_party_libs
        gdal
        arrow
        miniz
        )

set_property(TARGET arctern PROPERTY CUDA_STANDARD 14)
if (BUILD_WITH_GPU)
    set(third_party_libs
            ${third_party_libs}
            GL
            EGL
            cudart
            )
else ()
    set(third_party_libs
            ${third_party_libs}
            GLU
            OSMesa
            )
endif ()

target_link_libraries(arctern
            pthread
            ${third_party_libs}
)


install(TARGETS arctern
        DESTINATION lib)

