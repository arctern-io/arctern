include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(2d_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/scatter_plot/pointmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/choropleth_map/choropleth_map.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/general_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/heatmap/heatmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/heatmap/set_color.cpp
        )
set(window_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/window/window2d.cpp
        )

if (USE_GPU)
    set(window_src render/window/window_egl/window_gpu_2d.cpp
            render/2d/cuda/heatmap.cu)
else ()
    set(window_src render/window/window_osmesa/window_cpu_2d.cpp)
endif ()

set(util_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega_scatter_plot/vega_circle2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega_choropleth_map/vega_choropleth_map.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gis_functions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gdal/gis_functions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gdal/type_scan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/gdal/arctern_geometry.cpp
        )

set(LIBRARY_TYPE SHARED)
if (STATIC_LIB_SWITCH)
    set(LIBRARY_TYPE STATIC)
endif (STATIC_LIB_SWITCH)

add_library(
    windrose ${LIBRARY_TYPE}
    ${render_src}
    ${gis_src}
)

add_dependencies(windrose miniz_ep stb_ep)
target_link_libraries(windrose
	        gdal
        arrow
            pthread
            miniz
)


set_property(TARGET windrose PROPERTY CUDA_STANDARD 14)
if (USE_GPU)
    add_subdirectory(gis/cuda)
    target_link_libraries(windrose
            GL
            EGL
            cudart
            )
else ()
    target_link_libraries(windrose
            GLU
            OSMesa
            )
endif ()


install(TARGETS windrose
        DESTINATION lib)

