set(2d_src
        ${CMAKE_CURRENT_SOURCE_DIR}/2d/pointmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/2d/general_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/2d/render_builder.cpp
        )
set(window_src
        ${CMAKE_CURRENT_SOURCE_DIR}/window/window2d.cpp
        )

set(util_src
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/vega/vega_scatter_plot/vega_circle2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/color/color.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/vega/vega.cpp
        )

if (STATIC_LIB_SWITCH)
    add_library(GIS STATIC
            ${2d_src}
            ${window_src}
            ${util_src}
            )
else()
    add_library(GIS SHARED
            ${2d_src}
            ${window_src}
            ${util_src}
            )
endif(STATIC_LIB_SWITCH)

install(TARGETS GIS
        DESTINATION lib)

include_directories(${cuda_inc})
link_directories(${cuda_lib})

set_property(TARGET GIS PROPERTY CUDA_STANDARD 11)
add_dependencies(GIS miniz_ep arrow_ep)
target_link_libraries(GIS
        GL
        EGL
        tbb
        cudart
        arrow
        pthread
        miniz
        jemalloc_pic
        )

install(TARGETS GIS
        LIBRARY DESTINATION lib)
