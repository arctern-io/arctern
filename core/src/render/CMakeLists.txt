set(2d_src
        ${CMAKE_CURRENT_SOURCE_DIR}/2d/pointmap.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/2d/general_2d.cpp
        )
set(window_src
        ${CMAKE_CURRENT_SOURCE_DIR}/window/window2d.cpp
        )

if (STATIC_LIB_SWITCH)
    add_library(libGIS STATIC
            ${2d_src}
            ${window_src}
            )
else()
    add_library(libGIS SHARED
            ${2d_src}
            ${window_src}
            )
endif(STATIC_LIB_SWITCH)

install(TARGETS libGIS
        DESTINATION lib)

include_directories(${cuda_inc})
link_directories(${cuda_lib})

set_property(TARGET libGIS PROPERTY CUDA_STANDARD 11)

