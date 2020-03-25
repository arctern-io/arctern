#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
#set -o xtrace
# shellcheck disable=SC1091

# Load libraries
. /libspark.sh

# Load Spark environment variables
eval "$(spark_env)"

if [ ! $EUID -eq 0 ] && [ -e "/usr/lib/libnss_wrapper.so" ]; then
    echo "spark:x:$(id -u):$(id -g):Spark:$SPARK_HOME:/bin/false" > "$NSS_WRAPPER_PASSWD"
    echo "spark:x:$(id -g):" > "$NSS_WRAPPER_GROUP"
    echo "LD_PRELOAD=/usr/lib/libnss_wrapper.so" >> "$SPARK_CONFDIR/spark-env.sh"
fi

if [ -d "/opt/conda/envs/arctern" ];then
    echo "export GDAL_DATA=/opt/conda/envs/arctern/share/gdal" >> "$SPARK_CONFDIR/spark-env.sh"
    echo "export PROJ_LIB=/opt/conda/envs/arctern/share/proj" >> "$SPARK_CONFDIR/spark-env.sh"
    echo "export PYSPARK_PYTHON=/opt/conda/envs/arctern/bin/python" >> "$SPARK_CONFDIR/spark-env.sh"
fi

if [[ "$*" = "/run.sh" ]]; then
    info "** Starting Spark setup **"
    /setup.sh
    info "** Spark setup finished! **"
fi

echo ""
exec "$@"
