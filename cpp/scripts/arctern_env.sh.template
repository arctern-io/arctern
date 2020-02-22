#!/bin/bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

export ARCTERN_LIB_DIR="${SCRIPTS_DIR}/../lib"

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ARCTERN_LIB_DIR

if [ -n $CONDA_PREFIX ]; then
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

