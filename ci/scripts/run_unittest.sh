#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

HELP="
Usage:
  $0 [flags] [Arguments]

    -i [ARCTERN_INSTALL_PREFIX] or --install_prefix=[ARCTERN_INSTALL_PREFIX]
                              Install directory used by install.
    -e [CONDA_ENV] or --conda_env=[CONDA_ENV]
                              Setting conda activate environment
    -h or --help              Print help information


Use \"$0  --help\" for more information about a given command.
"

ARGS=`getopt -o "i:e:h" -l "install_prefix::,conda_env::,help" -n "$0" -- "$@"`

eval set -- "${ARGS}"

while true ; do
        case "$1" in
                -i|--install_prefix)
                        # o has an optional argument. As we are in quoted mode,
                        # an empty parameter will be generated if its optional
                        # argument is not found.
                        case "$2" in
                                "") echo "Option install_prefix, no argument"; exit 1 ;;
                                *)  ARCTERN_INSTALL_PREFIX=$2 ; shift 2 ;;
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
