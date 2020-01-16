#!/bin/bash -ex

# define necessary parameters here
REPO_NAME="libgis"
BUILD_TYPE="Debug"
BUILD_UNITTEST_OPT=""
BUILD_BENCHMARK_OPT=""
STATIC_LIB_SWITCH_OPT=""
ENCRYPT_LOG="off"
VERBOSE="1"
PARALLEL_LEVEL=${PARALLEL_LEVEL:="8"}


# define some dirs here
INSTALL_PREFIX=`pwd`/../${BUILD_DIR}
MANIFEST_CACHE_DIR=${ZDB_BUILD_PATH}/manifest_cache/${REPO_NAME}
MANIFEST_CACHE_FILE=${MANIFEST_CACHE_DIR}/install_manifest.txt


### user parameters defined begin
### user parameters defined end


# get command parameters
while getopts "t:e:hus" arg
do
        case $arg in
             t)
                BUILD_TYPE=$OPTARG # BUILD_TYPE
                ;;
             e)
                ENCRYPT_LOG=$OPTARG # ENCRYPT_LOG
                ;;
             u)
                echo "[INFO] building unittests" ;
                BUILD_UNITTEST_OPT="-DBUILD_UNIT_TEST=ON";
                ;;
             s)
                echo "[INFO] compile static lib" ;
                STATIC_LIB_SWITCH_OPT="-DSTATIC_LIB_SWITCH=ON";
                ;;
             h) # help
                echo "
parameter:
-t: build type [Release | Debug]
-e: cmake encrypt log
-u: building unit test options
-s: compile static lib
-h: help

usage:
./build.sh -t \${BUILD_TYPE} -e \${ENCRYPT_LOG} [-u] [-s] [-h] 
                "
                exit 0
                ;;
             ?)
                echo "[ERROR] unknown argument"
                exit 1
                ;;
        esac
done


## begin to build 
if [[ -d cmake_build ]]; then
        rm -rf cmake_build
fi

mkdir cmake_build
cd cmake_build

echo installing  ${REPO_NAME}to $INSTALL_PREFIX

if [[ -f ${MANIFEST_CACHE_FILE} ]];then
  cp ${MANIFEST_CACHE_FILE} .
fi

# prepare cmake_cmd
CMAKE_CMD="cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DCMAKE_ENCRYPT_LOG=${ENCRYPT_LOG} \
-DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
${BUILD_UNITTEST_OPT} \
${STATIC_LIB_SWITCH_OPT} \
../"

echo ${CMAKE_CMD}

${CMAKE_CMD}
make clean && make -j${PARALLEL_LEVEL}  VERBOSE=${VERBOSE} || exit 1
make uninstall && make  -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} install


# cache manifest list
if [[ -f ./install_manifest.txt ]];then
  if [[ ! -d ${MANIFEST_CACHE_DIR} ]];then
    mkdir -p ${MANIFEST_CACHE_DIR}
  fi
  cp ./install_manifest.txt ${MANIFEST_CACHE_FILE}
fi

echo build ${REPO_NAME} done!
