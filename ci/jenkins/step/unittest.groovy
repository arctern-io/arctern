timeout(time: 5, unit: 'MINUTES') {
    dir ("ci/scripts") {
        sh "/bin/bash --login -c './run_unittest.sh -e \"zgis_dev\"'"
        sh "/bin/bash --login -c './run_pytest.sh -e \"zgis_dev\"'"
    }
}
