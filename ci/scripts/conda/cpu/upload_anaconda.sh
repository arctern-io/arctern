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
    export ARCTERN_FILE=`conda build conda/recipes/arctern/cpu -c conda-forge -c defaults --output`
    LABEL_OPTION="--label main"
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
