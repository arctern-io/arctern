#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

cd ${SCRIPTS_DIR}

LIBARCTERN_FILE=${LIBARCTERN_FILE:="libarctern"}
ARCTERN_FILE=${ARCTERN_FILE:="arctern"}
ARCTERN_SPARK_FILE=${ARCTERN_SPARK_FILE:="arctern-spark"}

if [ -f ${SCRIPTS_DIR}/libarctern-[0-9]\.[0-9]\.[0-9]\.*tar* ];then
    LIBARCTERN_FILE=`echo libarctern-[0-9]\.[0-9]\.[0-9]\.*tar*`
fi

if [ -f ${SCRIPTS_DIR}/arctern-[0-9]\.[0-9]\.[0-9]\.*tar* ];then
    ARCTERN_FILE=`echo arctern-[0-9]\.[0-9]\.[0-9]\.*tar*`
fi

if [ -f ${SCRIPTS_DIR}/arctern-spark-[0-9]\.[0-9]\.[0-9]\.*tar* ];then
    ARCTERN_SPARK_FILE=`echo arctern-spark-[0-9]\.[0-9]\.[0-9]\.*tar*`
fi

conda install -y -q -n arctern -c conda-forge -c arctern-dev/label/cuda10.0 ${LIBARCTERN_FILE}
conda install -y -q -n arctern -c conda-forge -c arctern-dev ${ARCTERN_FILE} ${ARCTERN_SPARK_FILE}
conda clean --all -y
