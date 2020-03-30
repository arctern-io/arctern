#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

PYSPARK_SRC_DIR="${SCRIPTS_DIR}/../../spark/pyspark"

HELP="
Usage:
  $0 [flags] [Arguments]

    clean                     Remove all existing build artifacts and configuration (start over)
    -e [CONDA_ENV] or --conda_env=[CONDA_ENV]
                              Setting conda activate environment
    --library_dirs            Directories to search for external C libraries
    -h or --help              Print help information


Use \"$0  --help\" for more information about a given command.
"

ARGS=`getopt -o "e:h" -l "conda_env::,library_dirs::,help" -n "$0" -- "$@"`

eval set -- "${ARGS}"

while true ; do
        case "$1" in
                -e|--conda_env)
                        case "$2" in
                                "") echo "Option conda_env, no argument"; exit 1 ;;
                                *)  CONDA_ENV=$2 ; shift 2 ;;
                        esac ;;
                --library_dirs)
                        case "$2" in
                                "") echo "Option library_dirs, no argument"; exit 1 ;;
                                *)  LIBRARY_DIRS=$2 ; shift 2 ;;
                        esac ;;
                -h|--help) echo -e "${HELP}" ; exit 0 ;;
                --) shift ; break ;;
                *) echo "Internal error!" ; exit 1 ;;
        esac
done

# Set defaults for vars modified by flags to this script
CLEANUP=${CLEANUP:="OFF"}

if [[ -n ${CONDA_ENV} ]]; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
fi

pushd ${PYSPARK_SRC_DIR}

for arg do
if [[ $arg == "clean" ]];then
    if [[ -d ${PYSPARK_SRC_DIR}/build ]]; then
        rm -rf ${PYSPARK_SRC_DIR}/build
    fi
    if [[ -d ${PYSPARK_SRC_DIR}/dist ]]; then
        rm -rf ${PYSPARK_SRC_DIR}/dist
    fi
    rm -rf *.egg*
    exit 0
fi
done

if [[ ${CLEANUP} == "ON" ]];then
    if [[ -d ${PYSPARK_SRC_DIR}/build ]]; then
        rm -rf ${PYSPARK_SRC_DIR}/build
    fi
    if [[ -d ${PYSPARK_SRC_DIR}/dist ]]; then
        rm -rf ${PYSPARK_SRC_DIR}/dist
    fi
    rm -rf *.egg*
    exit 0
fi

if [[ -n ${LIBRARY_DIRS} ]];then
    python setup.py build build_ext --library-dirs=${LIBRARY_DIRS}
else
    python setup.py build
fi
python setup.py install

popd
