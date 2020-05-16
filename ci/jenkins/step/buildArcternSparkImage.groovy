dir ("docker/spark/${BINARY_VERSION}/${OS_NAME}/runtime") {
    def channelPackage = "conda-bld.tar.gz"
    def downloadStatus = sh(returnStatus: true, script: "curl -C - -o arctern/${channelPackage} ${ARTFACTORY_URL}/${channelPackage}")

    if (downloadStatus != 0) {
        error("\" Download \" ${ARTFACTORY_URL}/${channelPackage} \" failed!")
    }

    sh "tar zxvf arctern/${channelPackage} -C ./arctern"

    def baseImageName = "${ARCTERN_REPO}:${OS_NAME}-base"
    sh "docker pull ${baseImageName}"

    def imageName = "${params.DOKCER_REGISTRY_URL}/${REPO_NAME}:${TAG_NAME}"

    try {
        sh "docker build -t ${imageName} --build-arg IMAGE_NAME=${ARCTERN_REPO} ."
        try {
            withCredentials([usernamePassword(credentialsId: "${params.DOCKER_CREDENTIALS_ID}", usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                sh "docker login -u ${DOCKER_USERNAME} -p ${DOCKER_PASSWORD} ${params.DOKCER_REGISTRY_URL}"
                sh "docker push ${imageName}"
            }
        } catch (exc) {
            throw exc
        } finally {
            sh "docker logout ${params.DOKCER_REGISTRY_URL}"
        }
    } catch (exc) {
        throw exc
    } finally {
        deleteImages("${imageName}", true)
        sh(returnStatus: true, script: "docker rmi -f \$(docker images | grep '<none>' | awk '{print \$3}')")
    }
}

boolean deleteImages(String imageName, boolean force) {
    def imageNameStr = imageName.trim()
    def isExistImage = sh(returnStatus: true, script: "docker inspect --type=image ${imageNameStr} 2>&1 > /dev/null")
    if (isExistImage == 0) {
        def deleteImageStatus = 0
        if (force) {
            deleteImageStatus = sh(returnStatus: true, script: "docker rmi -f \$(docker inspect --type=image --format \"{{.ID}}\" ${imageNameStr}) 2>&1 > /dev/null")
        } else {
            deleteImageStatus = sh(returnStatus: true, script: "docker rmi ${imageNameStr}")
        }

        if (deleteImageStatus != 0) {
            return false
        }
    }
    return true
}
