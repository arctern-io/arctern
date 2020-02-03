include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(2d_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/pointmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/general_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/heatmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/set_color.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/cuda/heatmap.cu
        )
set(window_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/window/window2d.cpp
        )

if (CPU_ONLY)
    set(window_src render/window/window_osmesa/window_cpu_2d.cpp)
else ()
    set(window_src render/window/window_egl/window_gpu_2d.cpp)
endif ()

set(util_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega_scatter_plot/vega_circle2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega_heatmap/vega_heatmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/color/color.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/color/color_gradient.cpp
        )
set(arrow_src
        arrow/render_api.cpp
        )

set(render_src
        ${2d_src}
        ${window_src}
        ${util_src}
        ${arrow_src}
        )

set(gis_src
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/make_point.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/point_map.cpp
        )

set(gis_cpp_src
        gis/cpu/geometry.cpp
        )

set(LIBRARY_TYPE SHARED)
if (STATIC_LIB_SWITCH)
    set(LIBRARY_TYPE STATIC)
endif (STATIC_LIB_SWITCH)
add_library(
    GIS ${LIBRARY_TYPE}
    ${render_src}
    ${gis_src}
    ${gis_cpp_src}
)

add_library(
        zgeo SHARED
        ${gis_cpp_src}
)
add_dependencies(GIS miniz_ep)
add_dependencies(zgeo miniz_ep)

set_property(TARGET GIS PROPERTY CUDA_STANDARD 11)
if (CPU_ONLY)
    target_link_libraries(GIS
            GLU
            OSMesa
            arrow
            pthread
            miniz
            )
else ()
    target_link_libraries(GIS
            GL
            gdal
            EGL
            cudart
            arrow
            pthread
            miniz
            )
endif ()

target_link_libraries(zgeo
        arrow
        )

install(TARGETS GIS
        DESTINATION lib)

install(TARGETS zgeo
        DESTINATION lib)
