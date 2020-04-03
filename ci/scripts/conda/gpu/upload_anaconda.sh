#!/bin/bash
#
# Adopted from https://github.com/rapidsai/cudf/blob/branch-0.13/ci/cpu/upload_anaconda.sh

set -e

SOURCE_BRANCH=${SOURCE_BRANCH:=master}

# Restrict uploads to master branch
if [ "${GIT_BRANCH}" != "${SOURCE_BRANCH}" ]; then
    echo "Skipping upload"
    return 0
fi

if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
fi

if [ "$UPLOAD_LIBARCTERN" == "1" ]; then
    export LIBARCTERN_FILE=`conda build conda/recipes/libarctern/gpu -c conda-forge -c defaults -c nvidia --output`
    SPLIT_VERSION=(${CUDA_VERSION//./ })
    MINOR_VERSION=${SPLIT_VERSION[0]}.${SPLIT_VERSION[1]}
    LABEL_OPTION="--label cuda${MINOR_VERSION}"
    echo "LABEL_OPTION=${LABEL_OPTION}"

    test -e ${LIBARCTERN_FILE}
    echo "Upload libarctern..."
    echo ${LIBARCTERN_FILE}
    anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-arctern} ${LABEL_OPTION} --force ${LIBARCTERN_FILE}
fi

if [ "$UPLOAD_PYARCTERN" == "1" ]; then
    export PYARCTERN_FILE=`conda build conda/recipes/pyarctern -c conda-forge -c defaults --output`
    LABEL_OPTION="--label main"
    echo "LABEL_OPTION=${LABEL_OPTION}"

    test -e ${PYARCTERN_FILE}
    echo "Upload pyarctern..."
    echo ${PYARCTERN_FILE}
    anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-arctern} ${LABEL_OPTION} --force ${PYARCTERN_FILE}
fi

if [ "$UPLOAD_ARCTERN_SPARK" == "1" ]; then
    export ARCTERN_SPARK_FILE=`conda build conda/recipes/arctern-spark -c conda-forge -c defaults --output`
    LABEL_OPTION="--label main"
    echo "LABEL_OPTION=${LABEL_OPTION}"

    test -e ${ARCTERN_SPARK_FILE}
    echo "Upload arctern-spark..."
    echo ${ARCTERN_SPARK_FILE}
    anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-arctern} ${LABEL_OPTION} --force ${ARCTERN_SPARK_FILE}
fi
