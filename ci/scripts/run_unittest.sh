#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

while getopts "i:e:h" arg
do
        case $arg in
             i)
                ARCTERN_INSTALL_PREFIX=$OPTARG   # ARCTERN PATH
                ;;
             e)
                CONDA_ENV=$OPTARG # CONDA ENVIRONMENT
                ;;
             h) # help
                echo "

parameter:
-i: Arctern path
-e: set conda activate environment
-h: help

usage:
./run_unittest.sh -i \${ARCTERN_INSTALL_PREFIX} -e \${CONDA_ENV} [-h]
                "
                exit 0
                ;;
             ?)
                echo "ERROR! unknown argument"
        exit 1
        ;;
        esac
done

if [[ -n ${CONDA_ENV} ]]; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
fi

# Set defaults for vars that may not have been defined externally
#  FIXME: if ARCTERN_INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
ARCTERN_INSTALL_PREFIX=${ARCTERN_INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
ARCTERN_ENV_FILE=${ARCTERN_INSTALL_PREFIX}/scripts/arctern_env.sh

if [[ -f ${ARCTERN_ENV_FILE} ]];then
    source ${ARCTERN_ENV_FILE}
fi

ARCTERN_UNITTEST_DIR=${ARCTERN_INSTALL_PREFIX}/unittest

if [[ ! -d ${ARCTERN_UNITTEST_DIR} ]]; then
    echo "\"${ARCTERN_UNITTEST_DIR}\" directory does not exist !"
    exit 1
fi

for test in `ls ${ARCTERN_UNITTEST_DIR}`; do
    echo "run $test unittest"
    ${ARCTERN_UNITTEST_DIR}/${test}
    if [ $? -ne 0 ]; then
        echo "${ARCTERN_UNITTEST_DIR}/${test} run failed !"
        exit 1
    fi
done
