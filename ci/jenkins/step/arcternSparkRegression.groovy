timeout(time: 10, unit: 'MINUTES') {
    def composeProject = "${env.PIPELINE_NAME}-${env.BUILD_NUMBER}"

    if ("${BINARY_VERSION}" == "gpu") {
        composeProject = "${env.PIPELINE_NAME}-${BINARY_VERSION}-${env.BUILD_NUMBER}"
    }

    try {
        dir ("docker/test_env/spark/${BINARY_VERSION}") {
            sh "docker-compose -p ${composeProject} --compatibility up -d"
        }
        dir ("docker/test_env") {
            sh "docker-compose -p ${composeProject} --compatibility run --rm regression"
        }
    } catch(exc) {
        throw exc
    } finally {
        dir ("docker/test_env/spark/${BINARY_VERSION}") {
            sh "docker-compose -p ${composeProject} --compatibility down"
        }
    }
}
