 #!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

while getopts "i:b:e:h" arg
do
        case $arg in
             i)
                ARCTERN_INSTALL_PREFIX=$OPTARG   # ARCTERN INSTALL PATH
                ;;
             b)
                ARCTERN_BUILD_PREFIX=$OPTARG   # ARCTERN BUILD PATH
                ;;
             e)
                CONDA_ENV=$OPTARG # CONDA ENVIRONMENT
                ;;
             h) # help
                echo "
parameter:
-i: Arctern install path
-b: Arctern build path
-e: set conda activate environment
-h: help
usage:
./coverage.sh -i \${ARCTERN_INSTALL_PREFIX} -e \${CONDA_ENV} [-h]
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

ARCTERN_INSTALL_PREFIX=${ARCTERN_INSTALL_PREFIX}
ARCTERN_BUILD_PREFIX=${ARCTERN_BUILD_PREFIX}
ARCTERN_ENV_FILE=${ARCTERN_INSTALL_PREFIX}/scripts/arctern_env.sh

LCOV_CMD="lcov"
LCOV_GEN_CMD="genhtml"

FILE_INFO_BASE="base.info"
FILE_INFO_ARCTERN="server.info"
FILE_INFO_OUTPUT="output.info"
FILE_INFO_OUTPUT_NEW="output_new.info"
DIR_LCOV_OUTPUT="lcov_out"

if [[ -f ${ARCTERN_ENV_FILE} ]];then
    source ${ARCTERN_ENV_FILE}
fi

DIR_GCNO=${ARCTERN_BUILD_PREFIX}
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
    exit -1
fi

for test in `ls ${DIR_UNITTEST}`; do
    echo "run $test unittest"
    ${DIR_UNITTEST}/${test}
    if [ $? -ne 0 ]; then
        echo "${DIR_UNITTEST}/${test} run failed !"
        exit 1
    fi
done

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
    exit -2
fi
# gen html report
${LCOV_GEN_CMD} "${FILE_INFO_OUTPUT_NEW}" --output-directory ${DIR_LCOV_OUTPUT}/