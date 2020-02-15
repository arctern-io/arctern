#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

GIS_PATH="/var/lib/gis"

while getopts "i:e:h" arg
do
        case $arg in
             i)
                GIS_PATH=$OPTARG   # GIS PATH
                ;;
             e)
                CONDA_ENV=$OPTARG # CONDA ENVIRONMENT
                ;;
             h) # help
                echo "

parameter:
-i: GIS path
-e: set conda activate environment
-h: help

usage:
./run_unittest.sh -i \${GIS_PATH} -e \${CONDA_ENV} [-h]
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

GIS_UNITTEST_DIR=${GIS_PATH}/unittest

if [[ ! -d ${GIS_UNITTEST_DIR} ]]; then
    echo "\"${GIS_UNITTEST_DIR}\" directory does not exist !"
    exit 1
fi

env

for test in `ls ${GIS_UNITTEST_DIR}`; do
    echo "run $test unittest"
    ${GIS_UNITTEST_DIR}/${test}
    if [ $? -ne 0 ]; then
        echo "${GIS_UNITTEST_DIR}/${test} run failed !"
        exit 1
    fi
done
