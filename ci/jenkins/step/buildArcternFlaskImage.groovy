def baseImageName = "${ARCTERN_FLASK_REPO}:base"
sh "docker pull ${baseImageName}"

def imageName = "${FLASK_IMAGE_REPOSITORY}:${ARCTERN_FLASK_TAG}"

try {
    deleteImages("${imageName}", true)
    def customImage = docker.build("${imageName}", "--build-arg IMAGE_NAME=${ARCTERN_FLASK_REPO} --file docker/flask/runtime/Dockerfile .")
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
