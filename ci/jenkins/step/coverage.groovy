timeout(time: 5, unit: 'MINUTES') {
    dir ("ci/scripts") {
    	withCredentials([[$class: 'StringBinding', credentialsId: "arctern-ci-codecov-token", variable: 'CODECOV_TOKEN']]) {
            sh "/bin/bash --login -c './coverage.sh -e \"zgis_dev\"'"
        }
    }
}
