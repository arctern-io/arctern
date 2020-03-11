#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

LIBARCTERN_FILE=${LIBARCTERN_FILE:="libarctern"}
ARCTERN_FILE=${ARCTERN_FILE:="arctern"}
ARCTERN_SPARK_FILE=${ARCTERN_SPARK_FILE:="arctern-spark"}
ARCTERN_CHANNEL=${ARCTERN_CHANNEL:="arctern-dev"}

if [ -n "${CONDA_CHINA_CHANNEL}" ]; then
    conda config --add channels ${CONDA_CHINA_CHANNEL}
    conda config --set show_channel_urls yes
fi

cd ${SCRIPTS_DIR}

if [ -d ${SCRIPTS_DIR}/conda-bld ];then
conda install -y -q -n arctern -c conda-forge -c file://${SCRIPTS_DIR}/conda-bld ${LIBARCTERN_FILE} ${ARCTERN_FILE} ${ARCTERN_SPARK_FILE}
else
conda install -y -q -n arctern -c conda-forge -c ${ARCTERN_CHANNEL}/label/cuda10.0 ${LIBARCTERN_FILE}
conda install -y -q -n arctern -c conda-forge -c ${ARCTERN_CHANNEL} ${ARCTERN_FILE} ${ARCTERN_SPARK_FILE}
fi

conda clean --all -y
