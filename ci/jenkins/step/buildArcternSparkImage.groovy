dir ("docker/spark/${BINARY_VERSION}") {
    def channelPackage = "conda-bld.tar.gz"
    def downloadStatus = sh(returnStatus: true, script: "curl -C - -o ${channelPackage} ${ARTFACTORY_URL}/${channelPackage}")

    if (downloadStatus != 0) {
        error("\" Download \" ${ARTFACTORY_URL}/${channelPackage} \" failed!")
    }

    sh "tar zxvf ${channelPackage} -C ./${OS_NAME}/arctern"

    def sourceImage = "${ARCTERN_SPARK_SOURCE_REPO}:${ARCTERN_SPARK_SOURCE_TAG}"

    try {
        sh(returnStatus: true, script: "docker pull ${sourceImage}")
        sh "docker-compose build --force-rm spark-master"
        try {
            withCredentials([usernamePassword(credentialsId: "${params.DOCKER_CREDENTIALS_ID}", usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                sh "docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD} ${params.DOKCER_REGISTRY_URL}"
                sh "docker-compose push spark-master"
            }
        } catch (exc) {
            throw exc
        } finally {
            sh "docker logout ${params.DOKCER_REGISTRY_URL}"
        }
    } catch (exc) {
        throw exc
    } finally {
        deleteImages("${sourceImage}", true)
        sh "docker-compose down --rmi all"
        sh(returnStatus: true, script: "docker rmi -f \$(docker images | grep '<none>' | awk '{print \$3}')")
    }
}

boolean deleteImages(String imageName, boolean force) {
    def imageNameStr = imageName.trim()
    def deleteImageStatus = true
    if (force) {
        deleteImageStatus = sh(returnStatus: true, script: "docker rmi -f \$(docker inspect --type=image --format \"{{.ID}}\" ${imageNameStr}) 2>&1 > /dev/null")
    } else {
        deleteImageStatus = sh(returnStatus: true, script: "docker rmi ${imageNameStr}")
    }
    return deleteImageStatus
}
