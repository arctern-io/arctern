timeout(time: 15, unit: 'MINUTES') {
    dir ("ci/scripts") {
        if ("${BINARY_VERSION}" == "gpu") {
            sh "/bin/bash --login -c './cpp_build.sh -t ${params.BUILD_TYPE} -e \"zgis_dev\" -l -g -u --coverage'"
        } else {
            sh "/bin/bash --login -c './cpp_build.sh -t ${params.BUILD_TYPE} -e \"zgis_dev\" -l -u --coverage'"
        }
        sh "/bin/bash --login -c './python_build.sh -e \"zgis_dev\"'"
    }
}
