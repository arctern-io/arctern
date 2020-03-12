timeout(time: 30, unit: 'MINUTES') {
    dir ("ci/scripts/conda") {
        withCredentials([usernamePassword(credentialsId: "${params.JFROG_CREDENTIALS_ID}", usernameVariable: 'JFROG_USENAME', passwordVariable: 'UPLOAD_PACKAGE_FILE_KEY')]) {
	        if ("${BINARY_VERSION}" == "gpu") {
	            sh "/bin/bash --login -c 'source ./gpu/prebuild.sh && ./gpu/build.sh'"
	        } else {
	            sh "/bin/bash --login -c 'source ./cpu/prebuild.sh && ./cpu/build.sh'"
	        }
	    }
    }
}
