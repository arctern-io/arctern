#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

PYTHON_SRC_DIR="${SCRIPTS_DIR}/../../python"
ARCTERN_LIBRARY_DIRS=""

while getopts "l:e:h" arg
do
        case $arg in
             l)
                ARCTERN_LIBRARY_DIRS=$OPTARG   # ARCTERN LIBRARY DIRS
                ;;
             e)
                CONDA_ENV=$OPTARG # CONDA ENVIRONMENT
                ;;
             h) # help
                echo "

parameter:
-l: Arctern library directory
-e: set conda activate environment
-h: help

usage:
./python_build.sh -l \${ARCTERN_LIBRARY_DIRS} -e \${CONDA_ENV} [-h]
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

cd ${PYTHON_SRC_DIR}

if [[ -d ${PYTHON_SRC_DIR}/build ]]; then
    rm -rf ${PYTHON_SRC_DIR}/build
fi

if [[ -d ${PYTHON_SRC_DIR}/dist ]]; then
    rm -rf ${PYTHON_SRC_DIR}/dist
fi

rm -rf zilliz_gis.egg*

python setup.py build build_ext --library-dirs=${ARCTERN_LIBRARY_DIRS}
python setup.py install
