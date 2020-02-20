add_subdirectory(cpp)
if(USE_GPU)
    add_subdirectory(cuda)
endif()