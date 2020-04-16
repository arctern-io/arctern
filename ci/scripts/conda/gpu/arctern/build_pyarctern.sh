#!/usr/bin/env bash

set -e

if [ "$BUILD_ARCTERN" == '1' ]; then
  echo "Building arctern..."
  conda build conda/recipes/arctern -c defaults -c conda-forge
fi
