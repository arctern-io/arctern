timeout(time: 20, unit: 'MINUTES') {
    dir ("docker/spark/${BINARY_VERSION}/${OS_NAME}/runtime") {
        def channelPackage = "conda-bld.tar.gz"
        def downloadStatus = sh(returnStatus: true, script: "curl -C - -o arctern/${channelPackage} ${ARTFACTORY_URL}/${channelPackage}")

        if (downloadStatus != 0) {
            error("\" Download \" ${ARTFACTORY_URL}/${channelPackage} \" failed!")
        }

        sh "tar zxvf arctern/${channelPackage} -C ./arctern"

        def baseImageName = "${ARCTERN_REPO}:${OS_NAME}-base"
        sh "docker pull ${baseImageName}"

        def imageName = "${REPO_NAME}:${TAG_NAME}"

        try {
            deleteImages("${imageName}", true)
            def customImage = docker.build("${imageName}", "--build-arg IMAGE_NAME=${ARCTERN_REPO} .")
            deleteImages("${params.DOKCER_REGISTRY_URL}/${imageName}", true)
            docker.withRegistry("https://${params.DOKCER_REGISTRY_URL}", "${params.DOCKER_CREDENTIALS_ID}") {
                customImage.push()
            }
        } catch (exc) {
            throw exc
        } finally {
            deleteImages("${imageName}", true)
            deleteImages("${params.DOKCER_REGISTRY_URL}/${imageName}", true)
            deleteImages("${baseImageName}", true)
        }
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
