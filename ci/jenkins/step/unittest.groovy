timeout(time: 5, unit: 'MINUTES') {
    dir ("ci/scripts") {
        sh "/bin/bash --login -c 'source ${env.ARCTERN_INSTALL_PREFIX}/scripts/arctern_env.sh && ./run_unittest.sh -e \"zgis_dev\" -i ${env.ARCTERN_INSTALL_PREFIX}'"
        sh "/bin/bash --login -c 'source ${env.ARCTERN_INSTALL_PREFIX}/scripts/arctern_env.sh && ./run_pytest.sh -e \"zgis_dev\"'"
    }
}
