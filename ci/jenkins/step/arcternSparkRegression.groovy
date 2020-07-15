def composeProject = "${env.PIPELINE_NAME}-${SEMVER}-${OS_NAME}-${env.BUILD_NUMBER}".replaceAll("\\.", "-").replaceAll("_", "-")

if ("${BINARY_VERSION}" == "gpu") {
    composeProject = "${env.PIPELINE_NAME}-${SEMVER}-${OS_NAME}-${BINARY_VERSION}-${env.BUILD_NUMBER}".replaceAll("\\.", "-").replaceAll("_", "-")
}

try {
    dir ("docker/test_env/spark/${BINARY_VERSION}") {
        sh "docker-compose -p ${composeProject} --compatibility up -d"
    }
    dir ("docker/test_env") {
        sh "docker-compose -p ${composeProject} --compatibility run --rm regression"
        // sh "docker-compose pull --ignore-pull-failures restful-regression"
        // sh "docker-compose build restful-regression"
        // sh "docker-compose -p ${composeProject} --compatibility run --rm restful-regression"
        // if ("${SEMVER}" == "master") {
        //     withCredentials([usernamePassword(credentialsId: "${params.DOCKER_CREDENTIALS_ID}", usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        //         sh "docker login -u ${USERNAME} -p ${PASSWORD} ${params.DOKCER_REGISTRY_URL}"
        //         sh "docker-compose push --ignore-push-failures restful-regression"
        //         sh "docker logout ${params.DOKCER_REGISTRY_URL}"
        //     }
        // }
    }
} catch(exc) {
    dir ("docker/test_env/spark/${BINARY_VERSION}") {
        sh "docker-compose -p ${composeProject} logs spark-master"
        sh "docker-compose -p ${composeProject} logs spark-worker"
        sh "docker-compose -p ${composeProject} logs flask"
    }
    throw exc
} finally {
    dir ("docker/test_env/spark/${BINARY_VERSION}") {
        sh "docker-compose -p ${composeProject} --compatibility down --rmi all -v"
    }

    // dir ("docker/test_env") {
    //     sh "docker-compose -p ${composeProject} --compatibility down --rmi all -v"
    // }
}
