set(common_src
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/common/memory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/image/loader/image_loader.cpp
        )

set(layer_src
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/layer/layer_icon_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/layer/layer_circles_2d.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/engine/layer/layer_multi_color_circles_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/layer/layer_weighted_color_circles_2d.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/engine/layer/layer_building_weighted_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/layer/layer_weighted_pointsize_circles_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/layer/layer_weighted_circles_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/layer/layer_heatmap_2d.cu
        )

set(operator_src
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_icon_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_circles_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_general_2d.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_multi_color_circles_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_weighted_color_circles_2d.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_building_weighted_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_weighted_pointsize_circles_2d.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_weighted_circles_2d.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/operator_builder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_heatmap_2d.cpp
#        ${CMAKE_CURRENT_SOURCE_DIR}/engine/operator/op_cursor_inter.cpp
        )

set(window_src
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/window/window2d.cpp
        )

set(utils_src
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/color/color_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/color/color_gradient.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/vega/vega_parser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/dataset/dataset_accessor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/geo/geo_building_reader.cpp
        )

if (STATIC_LIB_SWITCH)
    add_library(libGIS STATIC
            ${layer_src}
            ${operator_src}
            ${window_src}
            ${utils_src}
            ${common_src}
            )
else()
    add_library(libGIS SHARED
            ${layer_src}
            ${operator_src}
            ${window_src}
            ${utils_src}
            ${common_src}
            )
endif(STATIC_LIB_SWITCH)

install(TARGETS libGIS
        DESTINATION lib)

include_directories(${cuda_inc})
link_directories(${cuda_lib})

set_property(TARGET libGIS PROPERTY CUDA_STANDARD 11)

