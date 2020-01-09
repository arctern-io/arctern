
macro(set_option_category name)
    set(GIS_OPTION_CATEGORY ${name})
    list(APPEND "GIS_OPTION_CATEGORIES" ${name})
endmacro()

macro(define_option name description default)
    option(${name} ${description} ${default})
    list(APPEND "GIS_${/home/zilliz/test_data/RawData}_OPTION_NAMES" ${name})
    set("${name}_OPTION_DESCRIPTION" ${description})
    set("${name}_OPTION_DEFAULT" ${default})
    set("${name}_OPTION_TYPE" "bool")
endmacro()

function(list_join lst glue out)
    if ("${${lst}}" STREQUAL "")
        set(${out} "" PARENT_SCOPE)
        return()
    endif ()

    list(GET ${lst} 0 joined)
    list(REMOVE_AT ${lst} 0)
    foreach (item ${${lst}})
        set(joined "${joined}${glue}${item}")
    endforeach ()
    set(${out} ${joined} PARENT_SCOPE)
endfunction()

macro(define_option_string name description default)
    set(${name} ${default} CACHE STRING ${description})
    list(APPEND "GIS_${GIS_OPTION_CATEGORY}_OPTION_NAMES" ${name})
    set("${name}_OPTION_DESCRIPTION" ${description})
    set("${name}_OPTION_DEFAULT" "\"${default}\"")
    set("${name}_OPTION_TYPE" "string")

    set("${name}_OPTION_ENUM" ${ARGN})
    list_join("${name}_OPTION_ENUM" "|" "${name}_OPTION_ENUM")
    if (NOT ("${${name}_OPTION_ENUM}" STREQUAL ""))
        set_property(CACHE ${name} PROPERTY STRINGS ${ARGN})
    endif ()
endmacro()

#----------------------------------------------------------------------
set_option_category("Gis Build Option")

define_option(GIS_GPU_VERSION "Build GPU version" OFF)
define_option(GIS_WITH_SQLITE "Build GPU version" AUTO)

#----------------------------------------------------------------------
set_option_category("Thirdparty")

set(GIS_DEPENDENCY_SOURCE_DEFAULT "BUNDLED")

define_option_string(GIS_DEPENDENCY_SOURCE
        "Method to use for acquiring GIS's build dependencies"
        "${GIS_DEPENDENCY_SOURCE_DEFAULT}"
        "AUTO"
        "BUNDLED"
        "SYSTEM")

define_option(GIS_USE_CCACHE "Use ccache when compiling (if available)" ON)

define_option(GIS_VERBOSE_THIRDPARTY_BUILD
        "Show output from ExternalProjects rather than just logging to files" ON)

#if (GIS_ENABLE_PROFILING STREQUAL "ON")
#    define_option(GIS_WITH_LIBUNWIND "Build with libunwind" ON)
#    define_option(GIS_WITH_GPERFTOOLS "Build with gperftools" ON)
#endif ()

#define_option(GIS_WITH_GRPC "Build with GRPC" ON)

#----------------------------------------------------------------------
set_option_category("Test and benchmark")

unset(GIS_BUILD_TESTS CACHE)
if (BUILD_UNIT_TEST)
    define_option(GIS_BUILD_TESTS "Build the GIS googletest unit tests" ON)
else ()
    define_option(GIS_BUILD_TESTS "Build the GIS googletest unit tests" OFF)
endif (BUILD_UNIT_TEST)

#----------------------------------------------------------------------
macro(config_summary)
    message(STATUS "---------------------------------------------------------------------")
    message(STATUS "GIS version:                                 ${GIS_VERSION}")
    message(STATUS)
    message(STATUS "Build configuration summary:")

    message(STATUS "  Generator: ${CMAKE_GENERATOR}")
    message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "  Source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
    if (${CMAKE_EXPORT_COMPILE_COMMANDS})
        message(
                STATUS "  Compile commands: ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
    endif ()

    foreach (category ${GIS_OPTION_CATEGORIES})

        message(STATUS)
        message(STATUS "${category} options:")

        set(option_names ${GIS_${category}_OPTION_NAMES})

        set(max_value_length 0)
        foreach (name ${option_names})
            string(LENGTH "\"${${name}}\"" value_length)
            if (${max_value_length} LESS ${value_length})
                set(max_value_length ${value_length})
            endif ()
        endforeach ()

        foreach (name ${option_names})
            if ("${${name}_OPTION_TYPE}" STREQUAL "string")
                set(value "\"${${name}}\"")
            else ()
                set(value "${${name}}")
            endif ()

            set(default ${${name}_OPTION_DEFAULT})
            set(description ${${name}_OPTION_DESCRIPTION})
            string(LENGTH ${description} description_length)
            if (${description_length} LESS 70)
                string(
                        SUBSTRING
                        "                                                                     "
                        ${description_length} -1 description_padding)
            else ()
                set(description_padding "
                ")
            endif ()

            set(comment "[${name}]")

            if ("${value}" STREQUAL "${default}")
                set(comment "[default] ${comment}")
            endif ()

            if (NOT ("${${name}_OPTION_ENUM}" STREQUAL ""))
                set(comment "${comment} [${${name}_OPTION_ENUM}]")
            endif ()

            string(
                    SUBSTRING "${value}                                                             "
                    0 ${max_value_length} value)

            message(STATUS "  ${description} ${description_padding} ${value} ${comment}")
        endforeach ()

    endforeach ()

endmacro()
