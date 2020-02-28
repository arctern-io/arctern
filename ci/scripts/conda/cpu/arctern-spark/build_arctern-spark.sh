#!/usr/bin/env bash

set -e

if [ "$BUILD_ARCTERN_SPARK" == '1' ]; then
  echo "Building Arctern Spark..."
  conda build conda/recipes/arctern-spark/cpu -c defaults -c conda-forge
fi
