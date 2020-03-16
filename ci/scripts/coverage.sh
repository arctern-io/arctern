#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

CPP_SRC_DIR="${SCRIPTS_DIR}/../../cpp"
CPP_BUILD_DIR="${CPP_SRC_DIR}/cmake_build"

HELP="
Usage:
  $0 [flags] [Arguments]

    --install_prefix=[ARCTERN_INSTALL_PREFIX]
                                      Install directory used by install.
    -e [CONDA_ENV] or --conda_env=[CONDA_ENV]
                                      Setting conda activate environment
    -n NAME                           Custom defined name of the upload. Visible in Codecov UI
    -F flag                           Flag the upload to group coverage metrics
    -h or --help                      Print help information


Use \"$0  --help\" for more information about a given command.
"

ARGS=`getopt -o "e:n:F:h" -l "install_prefix::,code_token::,help" -n "$0" -- "$@"`

eval set -- "${ARGS}"

while true ; do
        case "$1" in
                --install_prefix)
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
                -n)
                        case "$2" in
                                "") echo "Option codecov_name, no argument"; exit 1 ;;
                                *)  CODECOV_NAME=$2 ; shift 2 ;;
                        esac ;;
                -F)
                        case "$2" in
                                "") echo "Option codecov_flag, no argument"; exit 1 ;;
                                *)  CODECOV_FLAG=$2 ; shift 2 ;;
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

LCOV_CMD="lcov"
LCOV_GEN_CMD="genhtml"

FILE_INFO_BASE="base.info"
FILE_INFO_ARCTERN="server.info"
FILE_INFO_OUTPUT="output.info"
FILE_INFO_OUTPUT_NEW="output_new.info"
DIR_LCOV_OUTPUT="lcov_out"

DIR_GCNO=${CPP_BUILD_DIR}
DIR_UNITTEST=${ARCTERN_INSTALL_PREFIX}/unittest

# delete old code coverage info files
rm -f FILE_INFO_BASE
rm -f FILE_INFO_MILVUS
rm -f FILE_INFO_OUTPUT
rm -f FILE_INFO_OUTPUT_NEW
rm -rf lcov_out
rm -f FILE_INFO_BASE FILE_INFO_MILVUS FILE_INFO_OUTPUT FILE_INFO_OUTPUT_NEW

# get baseline
${LCOV_CMD} -c -i -d ${DIR_GCNO} -o "${FILE_INFO_BASE}"
if [ $? -ne 0 ]; then
    echo "gen baseline coverage run failed"
    exit 1
fi


# gen code coverage
${LCOV_CMD} -d ${DIR_GCNO} -o "${FILE_INFO_ARCTERN}" -c
# merge coverage
${LCOV_CMD} -a ${FILE_INFO_BASE} -a ${FILE_INFO_ARCTERN} -o "${FILE_INFO_OUTPUT}"

# remove third party from tracefiles
${LCOV_CMD} -r "${FILE_INFO_OUTPUT}" -o "${FILE_INFO_OUTPUT_NEW}" \
    "/usr/*" \
    "*/thirdparty/*" \
    "*/envs/*"

if [ $? -ne 0 ]; then
    echo "generate ${FILE_INFO_OUTPUT_NEW} failed"
    exit 2
fi

# gen html report
${LCOV_GEN_CMD} "${FILE_INFO_OUTPUT_NEW}" --output-directory ${DIR_LCOV_OUTPUT}/

if [ -n ${CODECOV_NAME} ];then
    CODECOV_NAME_OPTION=" -n ${CODECOV_NAME} "
fi

if [ -n ${CODECOV_FLAG} ];then
    CODECOV_FLAG_OPTION=" -F ${CODECOV_FLAG} "
fi

if [[ -n ${CODECOV_TOKEN} ]];then
    export CODECOV_TOKEN="${CODECOV_TOKEN}"
    curl -s https://codecov.io/bash | bash -s - -f ${FILE_INFO_OUTPUT_NEW} ${CODECOV_NAME_OPTION} ${CODECOV_FLAG_OPTION} || echo "Codecov did not collect coverage reports"
fi
