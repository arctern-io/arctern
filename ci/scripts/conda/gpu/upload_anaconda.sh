#!/bin/bash
#
# Adopted from https://github.com/rapidsai/cudf/blob/branch-0.13/ci/cpu/upload_anaconda.sh

set -e

BRANCH_REGEX='^(master|((v|branch\-)[0-9]+\.[0-9]+\.(x|[0-9]+|[0-9]+\-preview[0-9]*)))$'

# Restrict uploads to master branch
if [[ ! "${GIT_BRANCH}" =~ ${BRANCH_REGEX} ]]; then
    echo "Skipping upload"
    exit 0
fi

if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    exit 0
fi

if [ "$UPLOAD_ARCTERN" == "1" ]; then
    export ARCTERN_FILE=`conda build conda/recipes/arctern/gpu -c conda-forge -c defaults -c nvidia --output`
    SPLIT_VERSION=(${CUDA_VERSION//./ })
    MINOR_VERSION=${SPLIT_VERSION[0]}.${SPLIT_VERSION[1]}
    LABEL_OPTION="--label cuda${MINOR_VERSION}"
    echo "LABEL_OPTION=${LABEL_OPTION}"

    test -e ${ARCTERN_FILE}
    echo "Upload arctern..."
    echo ${ARCTERN_FILE}
    anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-arctern} ${LABEL_OPTION} --force ${ARCTERN_FILE}
fi

if [ "$UPLOAD_ARCTERN_WEBSERVER" == "1" ]; then
    export ARCTERN_WEBSERVER_FILE=`conda build conda/recipes/arctern-webserver -c conda-forge -c defaults --output`
    LABEL_OPTION="--label main"
    echo "LABEL_OPTION=${LABEL_OPTION}"

    test -e ${ARCTERN_WEBSERVER_FILE}
    echo "Upload arctern-webserver..."
    echo ${ARCTERN_WEBSERVER_FILE}
    anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-arctern} ${LABEL_OPTION} --force ${ARCTERN_WEBSERVER_FILE}
fi
