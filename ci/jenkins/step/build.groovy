timeout(time: 15, unit: 'MINUTES') {
    dir ("ci/scripts") {
        if ("${BINARY_VERSION}" == "gpu") {
            sh "/bin/bash --login -c './cpp_build.sh -t ${params.BUILD_TYPE} -o ${env.GIS_INSTALL_PREFIX} -e \"zgis_dev\" -g -u'"
        } else {
            sh "/bin/bash --login -c './cpp_build.sh -t ${params.BUILD_TYPE} -o ${env.GIS_INSTALL_PREFIX} -e \"zgis_dev\" -u'"
        }
        sh "/bin/bash --login -c 'source ${env.GIS_INSTALL_PREFIX}/scripts/gis_env.sh && ./python_build.sh -e \"zgis_dev\" -l \${GIS_LIB_DIR}'"
    }
}
