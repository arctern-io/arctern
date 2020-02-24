#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

ARCTERN_PATH="/var/lib/arctern"

while getopts "i:e:h" arg
do
        case $arg in
             i)
                ARCTERN_PATH=$OPTARG   # ARCTERN PATH
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
./run_unittest.sh -i \${ARCTERN_PATH} -e \${CONDA_ENV} [-h]
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
    if [[ -n ${CONDA_PREFIX} ]]; then
        LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_PREFIX}/lib
    fi
fi

ARCTERN_UNITTEST_DIR=${ARCTERN_PATH}/unittest

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
