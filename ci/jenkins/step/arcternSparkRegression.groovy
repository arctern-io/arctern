timeout(time: 10, unit: 'MINUTES') {
    try {
        dir ("docker/test_env/spark/${BINARY_VERSION}") {
            sh "docker-compose -p ${env.PIPELINE_NAME}-${env.BUILD_NUMBER} --compatibility up -d"
        }
        dir ("docker/test_env/spark/${BINARY_VERSION}") { 
            sh "docker-compose -p ${env.PIPELINE_NAME}-${env.BUILD_NUMBER} --compatibility run --rm regression"
        }
    } catch(exc) {
        throw exc
    } finally {
        sh "docker-compose -p ${env.PIPELINE_NAME}-${env.BUILD_NUMBER} --compatibility down"
    }
}
