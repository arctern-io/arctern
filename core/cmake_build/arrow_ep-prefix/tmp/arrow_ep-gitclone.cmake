
if(NOT "/home/czp/test/GIS/core/cmake_build/arrow_ep-prefix/src/arrow_ep-stamp/arrow_ep-gitinfo.txt" IS_NEWER_THAN "/home/czp/test/GIS/core/cmake_build/arrow_ep-prefix/src/arrow_ep-stamp/arrow_ep-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/czp/test/GIS/core/cmake_build/arrow_ep-prefix/src/arrow_ep-stamp/arrow_ep-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/home/czp/test/GIS/core/cmake_build/arrow/arrow"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/czp/test/GIS/core/cmake_build/arrow/arrow'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout "https://github.com/apache/arrow.git" "arrow"
    WORKING_DIRECTORY "/home/czp/test/GIS/core/cmake_build/arrow"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/apache/arrow.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout apache-arrow-0.15.1 --
  WORKING_DIRECTORY "/home/czp/test/GIS/core/cmake_build/arrow/arrow"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'apache-arrow-0.15.1'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/home/czp/test/GIS/core/cmake_build/arrow/arrow"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/czp/test/GIS/core/cmake_build/arrow/arrow'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/czp/test/GIS/core/cmake_build/arrow_ep-prefix/src/arrow_ep-stamp/arrow_ep-gitinfo.txt"
    "/home/czp/test/GIS/core/cmake_build/arrow_ep-prefix/src/arrow_ep-stamp/arrow_ep-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/czp/test/GIS/core/cmake_build/arrow_ep-prefix/src/arrow_ep-stamp/arrow_ep-gitclone-lastrun.txt'")
endif()

