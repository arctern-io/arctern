set(2d_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/pointmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/2d/general_2d.cpp
        )
set(window_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/window/window2d.cpp
        )

set(util_src
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega_scatter_plot/vega_circle2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/color/color.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/render/utils/vega/vega.cpp
        )

set(render_src
        ${2d_src}
        ${window_src}
        ${util_src}
)

set(gis_src
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/make_point.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/gis/point_map.cpp
)

set(gis_cpp_src
        gis/cpp/geometry/geometry.cpp
)

set(LIBRARY_TYPE SHARED)
if (STATIC_LIB_SWITCH)
    set(LIBRARY_TYPE STATIC)
endif(STATIC_LIB_SWITCH)
add_library(
    GIS ${LIBRARY_TYPE}
    ${render_src}
    ${gis_src}
)

add_library(
        zgeo SHARED
        ${gis_cpp_src}
)
add_dependencies(GIS miniz_ep)

set_property(TARGET GIS PROPERTY CUDA_STANDARD 11)
target_link_libraries(GIS
        GL
        EGL
        cudart
        arrow
        pthread
        miniz
        )

target_link_libraries(zgeo
        arrow
        )

install(TARGETS GIS
        DESTINATION lib)

install(TARGETS zgeo
        DESTINATION lib)
