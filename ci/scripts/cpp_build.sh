#!/bin/bash

set -e

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

    clean                     Remove all existing build artifacts and configuration (start over)
    -o [ARCTERN_INSTALL_PREFIX] or --install_prefix=[ARCTERN_INSTALL_PREFIX]
                              Install directory used by install.
    -t [BUILD_TYPE] or --build_type=[BUILD_TYPE]
                              Build type(default: Release)
    -e [CONDA_ENV] or --conda_env=[CONDA_ENV]
                              Setting conda activate environment
    -j[N] or --jobs=[N]       Allow N jobs at once; infinite jobs with no arg.
    -l                        Run cpplint & check clang-format
    -n                        No make and make install step
    -g                        Building for the architecture of the GPU in the system
    --coverage                Build Code Coverage
    -u or --tests             Build unittest case
    -p or --privileges        Install command with elevated privileges
    -v or --verbose           A level above ‘basic’; includes messages about which makefiles were parsed, prerequisites that did not need to be rebuilt
    -h or --help              Print help information


Use \"$0  --help\" for more information about a given command.
"

ARGS=`getopt -o "o:t:e:j::lngupvh" -l "install_prefix::,build_type::,conda_env::,coverage,tests,jobs::,privileges,help" -n "$0" -- "$@"`

eval set -- "${ARGS}"

while true ; do
        case "$1" in
                -o|--install_prefix)
                        # o has an optional argument. As we are in quoted mode,
                        # an empty parameter will be generated if its optional
                        # argument is not found.
                        case "$2" in
                                "") echo "Option install_prefix, no argument"; exit 1 ;;
                                *)  ARCTERN_INSTALL_PREFIX=$2 ; shift 2 ;;
                        esac ;;
                -t|--build_type)
                        case "$2" in
                                "") echo "Option build_type, no argument"; exit 1 ;;
                                *)  BUILD_TYPE=$2 ; shift 2 ;;
                        esac ;;
                -e|--conda_env)
                        case "$2" in
                                "") echo "Option conda_env, no argument"; exit 1 ;;
                                *)  CONDA_ENV=$2 ; shift 2 ;;
                        esac ;;
                -j|--jobs)
                        case "$2" in
                                "") PARALLEL_LEVEL=""; shift 2 ;;
                                *)  PARALLEL_LEVEL=$2 ; shift 2 ;;
                        esac ;;
                -g) echo "Building for the architecture of the GPU in the system..." ; USE_GPU="ON" ; shift ;;
                --coverage) echo "Build code coverage" ; BUILD_COVERAGE="ON" ; shift ;;
                -u|--tests) echo "Build unittest cases" ; BUILD_UNITTEST="ON" ; shift ;;
                -n) echo "No build and install step" ; COMPILE_BUILD="OFF" ; shift ;;
                -l) RUN_LINT="ON" ; shift ;;
                -p|--privileges) PRIVILEGES="ON" ; shift ;;
                -v|--verbose) VERBOSE="1" ; shift ;;
                -h|--help) echo -e "${HELP}" ; exit 0 ;;
                --) shift ; break ;;
                *) echo "Internal error!" ; exit 1 ;;
        esac
done

# Set defaults for vars modified by flags to this script
CUDA_COMPILER=/usr/local/cuda/bin/nvcc
VERBOSE=${VERBOSE:=""}
BUILD_TYPE=${BUILD_TYPE:="Release"}
BUILD_UNITTEST=${BUILD_UNITTEST:="OFF"}
BUILD_COVERAGE=${BUILD_COVERAGE:="OFF"}
COMPILE_BUILD=${COMPILE_BUILD:="ON"}
USE_GPU=${USE_GPU:="OFF"}
RUN_LINT=${RUN_LINT:="OFF"}
PRIVILEGES=${PRIVILEGES:="OFF"}
CLEANUP=${CLEANUP:="OFF"}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}

if [[ -n ${CONDA_ENV} ]]; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
fi

for arg do
if [[ $arg == "clean" ]];then
    echo "Remove all existing build artifacts and configuration..."
    if [ -d ${CPP_BUILD_DIR} ]; then
        find ${CPP_BUILD_DIR} -mindepth 1 -delete
        rmdir ${CPP_BUILD_DIR} || true
    fi
    exit 0
fi
done

# Set defaults for vars that may not have been defined externally
#  FIXME: if ARCTERN_INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
ARCTERN_INSTALL_PREFIX=${ARCTERN_INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}

echo -e "===\n=== ccache statistics before build\n==="
ccache --show-stats

if [[ ! -d ${CPP_BUILD_DIR} ]]; then
    mkdir ${CPP_BUILD_DIR}
fi

pushd ${CPP_BUILD_DIR}

CMAKE_CMD="cmake \
-DCMAKE_INSTALL_PREFIX=${ARCTERN_INSTALL_PREFIX} \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DBUILD_WITH_GPU=${USE_GPU} \
-DBUILD_UNITTEST=${BUILD_UNITTEST} \
-DBUILD_COVERAGE=${BUILD_COVERAGE} \
-DCMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
${CPP_SRC_DIR}
"
echo ${CMAKE_CMD}
${CMAKE_CMD}

if [[ ${RUN_LINT} == "ON" ]];then
    make lint || exit 1
    make check-clang-format || exit 1
fi

if [[ ${COMPILE_BUILD} == "ON" ]];then
    # compile and build
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} || exit 1

    if [[ ${PRIVILEGES} == "ON" ]];then
        sudo make install || exit 1
    else
        make install || exit 1
    fi
fi

popd
