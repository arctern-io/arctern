timeout(time: 5, unit: 'MINUTES') {
    dir ("ci/scripts") {
        sh "/bin/bash --login -c 'source ${env.GIS_INSTALL_PREFIX}/scripts/gis_env.sh && ./run_unittest.sh -e \"zgis_dev\" -i ${env.GIS_INSTALL_PREFIX}'"
        sh "/bin/bash --login -c 'source ${env.GIS_INSTALL_PREFIX}/scripts/gis_env.sh && ./run_pytest.sh"
    }
}
