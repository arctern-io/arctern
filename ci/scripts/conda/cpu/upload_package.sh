#!/bin/bash

set -e

if [ -z "$UPLOAD_PACKAGE_FILE_KEY" ]; then
    echo "No upload package file key"
    return 0
fi

if [ "$UPLOAD_LIBARCTERN" == "1" ]; then
    export LIBARCTERN_FILE=`conda build conda/recipes/libarctern/cpu -c conda-forge -c defaults --output`

    test -e ${LIBARCTERN_FILE}
    echo "Upload libarctern package file ..."
    echo ${LIBARCTERN_FILE}
    curl -u${JFROG_USENAME:-arctern}:${UPLOAD_PACKAGE_FILE_KEY} -T ${LIBARCTERN_FILE} ${ARTFACTORY_URL}/${LIBARCTERN_FILE##*/}
fi

if [ "$UPLOAD_ARCTERN" == "1" ]; then
    export ARCTERN_FILE=`conda build conda/recipes/arctern -c conda-forge -c defaults --output`

    test -e ${ARCTERN_FILE}
    echo "Upload arctern package file ..."
    echo ${ARCTERN_FILE}
    curl -u${JFROG_USENAME:-arctern}:${UPLOAD_PACKAGE_FILE_KEY} -T ${ARCTERN_FILE} ${ARTFACTORY_URL}/${ARCTERN_FILE##*/}
fi

if [ "$UPLOAD_ARCTERN_SPARK" == "1" ]; then
    export ARCTERN_SPARK_FILE=`conda build conda/recipes/arctern-spark -c conda-forge -c defaults --output`

    test -e ${ARCTERN_SPARK_FILE}
    echo "Upload arctern-spark package file..."
    echo ${ARCTERN_SPARK_FILE}
    curl -u${JFROG_USENAME:-arctern}:${UPLOAD_PACKAGE_FILE_KEY} -T ${ARCTERN_SPARK_FILE} ${ARTFACTORY_URL}/${ARCTERN_SPARK_FILE##*/}
fi
