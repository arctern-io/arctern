#!/usr/bin/env bash

set -e

if [ "$BUILD_ARCTERN_WEBSERVER" == '1' ]; then
    echo "Building arctern-webserver..."
    conda build conda/recipes/arctern-webserver -c defaults -c conda-forge
fi
