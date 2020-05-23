#!/bin/bash

set -e

if [ -z "$UPLOAD_PACKAGE_FILE_KEY" ]; then
    echo "No upload package file key"
    return 0
fi

if [[ "$UPLOAD_ARCTERN" == "1" || "$UPLOAD_ARCTERN_WEBSERVER" == "1" ]]; then
    if [ -d ${CONDA_PREFIX}/conda-bld ];then
        tar -zcf ./conda-bld.tar.gz -C ${CONDA_PREFIX}/ conda-bld
        curl -u${JFROG_USENAME:-arctern}:${UPLOAD_PACKAGE_FILE_KEY} -T ./conda-bld.tar.gz ${ARTFACTORY_URL}/conda-bld.tar.gz
        rm ./conda-bld.tar.gz
    fi
fi
