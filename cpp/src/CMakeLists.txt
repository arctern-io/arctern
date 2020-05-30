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
        render/2d/fishnet_map/fishnet_map.cpp
        render/2d/scatter_plot/pointmap.cpp
        render/2d/scatter_plot/weighted_pointmap.cpp
        render/2d/choropleth_map/choropleth_map.cpp
        render/2d/general_2d.cpp
        render/2d/heatmap/heatmap.cpp
        render/2d/heatmap/set_color.cpp
        render/2d/icon/icon_viz.cpp
        render/utils/vega/vega_fishnet_map/vega_fishnet_map.cpp
        render/utils/render_utils.cpp
        render/utils/agg/agg_handler.cpp
        render/utils/vega/vega_scatter_plot/vega_pointmap.cpp
        render/utils/vega/vega_scatter_plot/vega_weighted_pointmap.cpp
        render/utils/vega/vega_choropleth_map/vega_choropleth_map.cpp
        render/utils/vega/vega.cpp
        render/utils/vega/vega_heatmap/vega_heatmap.cpp
        render/utils/vega/vega_icon/vega_icon.cpp
        render/utils/color/color.cpp
        render/utils/color/color_gradient.cpp
        render/utils/image/image_loader.cpp
        )

set(cpu_render_src
        render/window/window_osmesa/window_cpu_2d.cpp
        )

set(gpu_render_src
        render/window/window_egl/window_gpu_2d.cpp
        render/2d/cuda/heatmap.cu
        )

set(gis_src
        gis/gis_functions.cpp
        gis/gdal/gis_functions.cpp
        gis/parser.cpp
        gis/gdal/geometry_visitor.cpp
        gis/dispatch/wkt_type_scanner.cpp
        gis/dispatch/wkb_type_scanner.cpp
        gis/dispatch/type_scanner.cpp
        gis/gdal/format_conversion.cpp
        gis/spatial_join/st_indexed_within.cpp
        )

set(cuda_gis_src
        gis/cuda/conversion/arrow_decode.cu
        gis/cuda/conversion/arrow_decode.cpp
        gis/cuda/conversion/arrow_encode.cu
        gis/cuda/conversion/arrow_encode.cpp
        gis/cuda/common/gis_definitions.cpp
        gis/cuda/common/gis_definitions.cu
        gis/cuda/functor/st_distance.cu
        gis/cuda/functor/st_within.cu
        gis/cuda/functor/st_intersects.cu
        gis/cuda/functor/st_crosses.cu
        gis/cuda/functor/st_overlaps.cu
        gis/cuda/functor/st_touches.cu
        gis/cuda/functor/st_equals.cu
        gis/cuda/functor/st_length.cu
        gis/cuda/functor/st_area.cu
        gis/cuda/functor/st_envelope.cu
        gis/cuda/functor/st_point.cu
        gis/cuda/functor/st_relate.cu
        gis/cuda/gis_functions.cpp
        )

set(arrow_src
        arrow/render_api.cpp
        )

set(index_src
        index/index_builder.cpp
        )

set(map_match_src
        map_match/map_match.cpp)

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
            ${index_src}
            ${map_match_src}
    )
else ()
    add_library(
            arctern ${LIBRARY_TYPE}
            ${gis_src}
            ${render_src}
            ${cpu_render_src}
            ${arrow_src}
            ${index_src}
            ${map_match_src}
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
            ${CUDA_LIBRARIES}
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

