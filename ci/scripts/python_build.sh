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

HELP="
Usage:
  $0 [flags] [Arguments]

    clean                     Remove all existing build artifacts and configuration (start over)
    -i [INSTALL_PREFIX] or --install_prefix=[INSTALL_PREFIX]
                              Install prefix
    --arctern_install_prefix=[ARCTERN_INSTALL_PREFIX]
                              Arctern install directory used by install.
    -e [CONDA_ENV] or --conda_env=[CONDA_ENV]
                              Setting conda activate environment
    --library_dirs            Directories to search for external C libraries
    -h or --help              Print help information


Use \"$0  --help\" for more information about a given command.
"

ARGS=`getopt -o "i:e:h" -l "install_prefix::,conda_env::,library_dirs::,arctern_install_prefix::,help" -n "$0" -- "$@"`

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
                --arctern_install_prefix)
                        case "$2" in
                                "") echo "Option arctern_install_prefix, no argument"; exit 1 ;;
                                *)  ARCTERN_INSTALL_PREFIX=$2 ; shift 2 ;;
                        esac ;;
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

# Set defaults for vars that may not have been defined externally
#  FIXME: if ARCTERN_INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
ARCTERN_INSTALL_PREFIX=${ARCTERN_INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
ARCTERN_ENV_FILE=${ARCTERN_INSTALL_PREFIX}/scripts/arctern_env.sh

if [[ -f ${ARCTERN_ENV_FILE} ]];then
    source ${ARCTERN_ENV_FILE}
fi

pushd ${PYTHON_SRC_DIR}

for arg do
if [[ $arg == "clean" ]];then
    if [[ -d ${PYTHON_SRC_DIR}/build ]]; then
        rm -rf ${PYTHON_SRC_DIR}/build
    fi
    if [[ -d ${PYTHON_SRC_DIR}/dist ]]; then
        rm -rf ${PYTHON_SRC_DIR}/dist
    fi
    rm -rf *.egg*
    exit 0
fi
done

if [[ ${CLEANUP} == "ON" ]];then
    if [[ -d ${PYTHON_SRC_DIR}/build ]]; then
        rm -rf ${PYTHON_SRC_DIR}/build
    fi
    if [[ -d ${PYTHON_SRC_DIR}/dist ]]; then
        rm -rf ${PYTHON_SRC_DIR}/dist
    fi
    rm -rf *.egg*
    exit 0
fi

if [[ -n ${LIBRARY_DIRS} ]];then
    python setup.py build build_ext --library-dirs=${LIBRARY_DIRS}
else
    python setup.py build build_ext
fi

if [[ -n ${INSTALL_PREFIX} ]];then
    python setup.py install --prefix=${INSTALL_PREFIX}
else
    python setup.py install
fi

popd
