timeout(time: 30, unit: 'MINUTES') {
    dir ("ci/scripts/conda") {
    	withCredentials([[$class: 'StringBinding', credentialsId: "arctern-dev-anaconda-token", variable: 'MY_UPLOAD_KEY']]) {
	        if ("${BINARY_VERSION}" == "gpu") {
	            sh "/bin/bash --login -c 'source ./gpu/prebuild.sh && ./gpu/build.sh'"
	        } else {
	            sh "/bin/bash --login -c 'source ./cpu/prebuild.sh && ./cpu/build.sh'"
	        }
	    }
    }
}
