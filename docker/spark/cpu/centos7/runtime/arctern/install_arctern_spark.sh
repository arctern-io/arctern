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
PYARCTERN_FILE=${PYARCTERN_FILE:="arctern"}
ARCTERN_SPARK_FILE=${ARCTERN_SPARK_FILE:="arctern-spark"}
ARCTERN_CHANNEL=${ARCTERN_CHANNEL:="arctern-dev"}

if [ -n "${CONDA_CUSTOM_CHANNEL}" ]; then
    conda config --add channels ${CONDA_CUSTOM_CHANNEL}
    conda config --set show_channel_urls yes
    conda config --show channels
fi

cd ${SCRIPTS_DIR}

if [ -d ${SCRIPTS_DIR}/conda-bld ];then
conda install -y -q -n arctern -c conda-forge -c file://${SCRIPTS_DIR}/conda-bld ${PYARCTERN_FILE} ${ARCTERN_SPARK_FILE}
else
conda install -y -q -n arctern -c conda-forge -c ${ARCTERN_CHANNEL} ${PYARCTERN_FILE} ${ARCTERN_SPARK_FILE}
fi

conda install -y -q -n arctern -c conda-forge pyyaml shapely
conda clean --all -y
