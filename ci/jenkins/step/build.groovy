timeout(time: 15, unit: 'MINUTES') {
    def checkResult = sh(script: "./check_ccache.sh -l ${params.JFROG_ARTFACTORY_URL}/ccache", returnStatus: true)

    dir ("ci/scripts") {
        if ("${BINARY_VERSION}" == "gpu") {
            sh "/bin/bash --login -c './cpp_build.sh -t ${params.BUILD_TYPE} -e \"zgis_dev\" -l -g -u --coverage'"
        } else {
            sh "/bin/bash --login -c './cpp_build.sh -t ${params.BUILD_TYPE} -e \"zgis_dev\" -l -u --coverage'"
        }
        sh "/bin/bash --login -c './python_build.sh -e \"zgis_dev\"'"
    }

    withCredentials([usernamePassword(credentialsId: "${params.JFROG_CREDENTIALS_ID}", usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        def updateResult = sh(script: "./update_ccache.sh -l ${params.JFROG_ARTFACTORY_URL}/ccache -u ${USERNAME} -p ${PASSWORD}", returnStatus: true)
    }
}
