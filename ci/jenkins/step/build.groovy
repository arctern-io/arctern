timeout(time: 15, unit: 'MINUTES') {
    dir ("ci/scripts") {
        if ("${BINARY_VERSION}" == "gpu") {
            sh "/bin/bash --login -c './cpp_build.sh -t ${params.BUILD_TYPE} -o ${env.ARCTERN_INSTALL_PREFIX} -e \"zgis_dev\" -g -u'"
        } else {
            sh "/bin/bash --login -c './cpp_build.sh -t ${params.BUILD_TYPE} -o ${env.ARCTERN_INSTALL_PREFIX} -e \"zgis_dev\" -u'"
        }
        sh "/bin/bash --login -c 'source ${env.ARCTERN_INSTALL_PREFIX}/scripts/arctern_env.sh && ./python_build.sh -e \"zgis_dev\" -l \${ARCTERN_LIB_DIR}'"
    }
}
