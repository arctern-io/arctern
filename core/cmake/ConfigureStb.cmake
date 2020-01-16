set(STB_SOURCE          ${CMAKE_BINARY_DIR}/stb_ep-prefix/src/stb_ep)
set(STB_DESTINATION     ${CMAKE_BINARY_DIR}/thirdparty/include/stb)


file(MAKE_DIRECTORY     ${STB_DESTINATION})

set(STB_INSTALL_COMMAND cd ${STB_SOURCE} && ls | grep .h | xargs cp -t ${STB_DESTINATION})

ExternalProject_Add(stb_ep
                    GIT_REPOSITORY              https://github.com/nothings/stb
                    GIT_TAG                     master
                    CONFIGURE_COMMAND           ""
                    BUILD_COMMAND               ""
                    INSTALL_COMMAND             ${STB_INSTALL_COMMAND})