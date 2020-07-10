#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

ARCTERN_SPARK_SRC_DIR="${SCRIPTS_DIR}/../../spark"

HELP="
Usage:
  $0 [flags] [Arguments]

    clean                     Remove all existing build artifacts and configuration (start over)
    -i [INSTALL_PREFIX] or --install_prefix=[INSTALL_PREFIX]
                              Install prefix
    -e [CONDA_ENV] or --conda_env=[CONDA_ENV]
                              Setting conda activate environment


Use \"$0  --help\" for more information about a given command.
"

ARGS=`getopt -o "e:h" -l "install_prefix::,conda_env::,help" -n "$0" -- "$@"`

eval set -- "${ARGS}"

while true ; do
        case "$1" in
                -i|--install_prefix)
                        # o has an optional argument. As we are in quoted mode,
                        # an empty parameter will be generated if its optional
                        # argument is not found.
                        case "$2" in
                                "") echo "Option install_prefix, no argument"; exit 1 ;;
                                *)  INSTALL_PREFIX=$2 ; shift 2 ;;
                        esac ;;
                -e|--conda_env)
                        case "$2" in
                                "") echo "Option conda_env, no argument"; exit 1 ;;
                                *)  CONDA_ENV=$2 ; shift 2 ;;
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

ARCTERN_SCALA_SRC_DIR="${SCRIPTS_DIR}/../../scala"
pushd ${ARCTERN_SCALA_SRC_DIR}
# sbt clean assembly
sbt "set test in assembly := {}" clean assembly # skip test
mkdir -p ${CONDA_PREFIX}/jars
mv target/scala-2.12/*.jar ${CONDA_PREFIX}/jars/
popd

pushd ${ARCTERN_SPARK_SRC_DIR}

for arg do
if [[ $arg == "clean" ]];then
    if [[ -d ${ARCTERN_SPARK_SRC_DIR}/build ]]; then
        rm -rf ${ARCTERN_SPARK_SRC_DIR}/build
    fi
    if [[ -d ${ARCTERN_SPARK_SRC_DIR}/dist ]]; then
        rm -rf ${ARCTERN_SPARK_SRC_DIR}/dist
    fi
    rm -rf *.egg*
    exit 0
fi
done

if [[ ${CLEANUP} == "ON" ]];then
    if [[ -d ${ARCTERN_SPARK_SRC_DIR}/build ]]; then
        rm -rf ${ARCTERN_SPARK_SRC_DIR}/build
    fi
    if [[ -d ${ARCTERN_SPARK_SRC_DIR}/dist ]]; then
        rm -rf ${ARCTERN_SPARK_SRC_DIR}/dist
    fi
    rm -rf *.egg*
    exit 0
fi

if [[ -n ${INSTALL_PREFIX} ]];then
    python setup.py install --prefix=${INSTALL_PREFIX}
else
    python setup.py install
fi

popd

