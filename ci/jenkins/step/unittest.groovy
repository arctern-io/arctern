timeout(time: 5, unit: 'MINUTES') {
    dir ("ci/scripts") {
        sh "/bin/bash --login -c './run_unittest.sh -e \"arctern\"'"
        sh "/bin/bash --login -c './run_pytest.sh -e \"arctern\"'"
    }
}
