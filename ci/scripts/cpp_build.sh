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
BUILD_TYPE="Debug"
BUILD_UNITTEST="OFF"
INSTALL_PREFIX="/var/lib/arctern"
RUN_LINT="OFF";
COMPILE_BUILD="ON"
USE_GPU="OFF"
CUDA_COMPILER=/usr/local/cuda/bin/nvcc
PRIVILEGES="OFF"

while getopts "o:t:d:e:lnguph" arg
do
        case $arg in
             o)
                INSTALL_PREFIX=$OPTARG
                ;;
             t)
                BUILD_TYPE=$OPTARG # BUILD TYPE
                ;;
             d)
                CPP_BUILD_DIR=$OPTARG # CPP BUILD DIRCETORY
                ;;
             e)
                CONDA_ENV=$OPTARG # CONDA ENVIRONMENT
                ;;
             l)
                RUN_LINT="ON";
                ;;
             n)
                COMPILE_BUILD="OFF";
                ;;
             g)
                USE_GPU="ON";
                ;;
             u)
                echo "Build unittest cases" ;
                BUILD_UNITTEST="ON";
                ;;
             p)
                PRIVILEGES="ON" # ELEVATED PRIVILEGES
                ;;
             h) # help
                echo "

parameter:
-o: install prefix(default: /var/lib/arctern)
-t: build type(default: Debug)
-d: cpp code build directory
-e: set conda activate environment
-l: run cpplint & check clang-format
-n: no execute make and make install
-g: gpu version
-u: building unit test options(default: OFF)
-p: install command with elevated privileges
-h: help

usage:
./cpp_build.sh -o \${INSTALL_PREFIX} -t \${BUILD_TYPE} -d \${CPP_BUILD_DIR} -e \${CONDA_ENV} [-l] [-n] [-g] [-u] [-p] [-h]
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

if [[ ! -d ${CPP_BUILD_DIR} ]]; then
    mkdir ${CPP_BUILD_DIR}
fi

cd ${CPP_BUILD_DIR}

CMAKE_CMD="cmake \
-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DUSE_GPU=${USE_GPU} \
-DBUILD_UNITTEST=${BUILD_UNITTEST} \
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
    make -j8 || exit 1

    if [[ ${PRIVILEGES} == "ON" ]];then
        sudo make install || exit 1
    else
        make install || exit 1
    fi
fi

