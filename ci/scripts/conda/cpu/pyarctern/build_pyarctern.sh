#!/usr/bin/env bash

set -e

if [ "$BUILD_ARCTERN" == '1' ]; then
  echo "Building PyArctern..."
  conda build conda/recipes/pyarctern -c defaults -c conda-forge
fi
