
if(NOT "/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep-stamp/stb_ep-gitinfo.txt" IS_NEWER_THAN "/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep-stamp/stb_ep-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep-stamp/stb_ep-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout "https://github.com/nothings/stb" "stb_ep"
    WORKING_DIRECTORY "/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/nothings/stb'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout master --
  WORKING_DIRECTORY "/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'master'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep-stamp/stb_ep-gitinfo.txt"
    "/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep-stamp/stb_ep-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/czp/test/GIS/core/cmake_build/stb_ep-prefix/src/stb_ep-stamp/stb_ep-gitclone-lastrun.txt'")
endif()

