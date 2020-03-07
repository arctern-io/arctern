#!/bin/bash
set -e

MEGA_HOME=/home/arctern_test

echo 'docker given group is: '$GROUP
echo 'docker given user is: '$USER

if [ "$(id -u)" = "0" ]; then
    groupadd -f -g $GROUP $USER
    
    useradd -u $USER -m -r -g $GROUP $USER
    
    # chown $USER:$GROUP -R /megawise
    echo "switch user"
    su - $USER "$BASH_SOURCE"
fi

echo "ccurrent uid:" `id -u`
echo "ccurrent gid:" `id -g`


set -o errexit
set -o nounset
set -o pipefail
#set -o xtrace
# shellcheck disable=SC1091

# Load libraries
. /libspark.sh

# Load Spark environment variables
eval "$(spark_env)"

if [ ! $EUID -eq 0 ] && [ -e /usr/lib/libnss_wrapper.so ]; then
    echo "spark:x:$(id -u):$(id -g):Spark:$SPARK_HOME:/bin/false" > "$NSS_WRAPPER_PASSWD"
    echo "spark:x:$(id -g):" > "$NSS_WRAPPER_GROUP"
    echo "LD_PRELOAD=/usr/lib/libnss_wrapper.so" >> "$SPARK_CONFDIR/spark-env.sh"
fi

if [[ "$*" = "/run.sh" ]]; then
    info "** Starting Spark setup **"
    /setup.sh
    info "** Spark setup finished! **"
fi

echo ""
exec "$@"
