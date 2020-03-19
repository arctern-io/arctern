timeout(time: 5, unit: 'MINUTES') {
    dir ("ci/scripts") {
    	withCredentials([[$class: 'StringBinding', credentialsId: "arctern-ci-codecov-token", variable: 'CODECOV_TOKEN']]) {
            sh "/bin/bash --login -c './coverage.sh -e \"arctern\" -n ${CPU_ARCH}-linux-${BINARY_VERSION}-unittest -F ${CPU_ARCH}_linux_${BINARY_VERSION}_unittest '"
        }
    }
}
