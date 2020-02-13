timeout(time: 15, unit: 'MINUTES') {
    dir ("ci/scripts") {
        if ("${BINARY_VERSION}" == "gpu") {
            sh "/bin/bash --login -c \"./cpp_build.sh -t ${params.BUILD_TYPE} -o ${env.GIS_INSTALL_PREFIX} -g -u\""
        } else {
            sh "/bin/bash --login -c \"./cpp_build.sh -t ${params.BUILD_TYPE} -o ${env.GIS_INSTALL_PREFIX} -u\""
        }
    }
}
