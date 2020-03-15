timeout(time: 10, unit: 'MINUTES') {
    def composeProject = "${env.PIPELINE_NAME}-${SEMVER}-${env.BUILD_NUMBER}".replaceAll("\\.", "-").replaceAll("_", "-")

    if ("${BINARY_VERSION}" == "gpu") {
        composeProject = "${env.PIPELINE_NAME}-${SEMVER}-${BINARY_VERSION}-${env.BUILD_NUMBER}".replaceAll("\\.", "-").replaceAll("_", "-")
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
            sh "docker-compose -p ${composeProject} --compatibility down --rmi all -v"
        }

        dir ("docker/test_env") {
            sh "docker-compose -p ${composeProject} --compatibility down --rmi all -v"
        }
    }
}
