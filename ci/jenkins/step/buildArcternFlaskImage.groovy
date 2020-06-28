def imageName = "${params.DOKCER_REGISTRY_URL}/${FLASK_IMAGE_REPOSITORY}:${ARCTERN_FLASK_TAG}"

def cacheImageName = "${params.DOKCER_REGISTRY_URL}/${FLASK_IMAGE_REPOSITORY}:${ARCTERN_FLASK_TARGET_TAG}"

try {
    sh(returnStatus: true, script: "docker pull ${cacheImageName}")
    sh "docker build -t ${imageName} --cache-from ${cacheImageName} --file docker/flask/Dockerfile ."
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
    deleteImages("${cacheImageName}", true)
    sh(returnStatus: true, script: "docker rmi -f \$(docker images | grep '<none>' | awk '{print \$3}')")
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
