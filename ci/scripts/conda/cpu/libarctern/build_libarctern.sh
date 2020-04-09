#!/usr/bin/env bash

set -e

if [ "$BUILD_LIBARCTERN" == '1' ]; then
  echo "Building libarctern..."
  conda build conda/recipes/libarctern/cpu -c defaults -c conda-forge -c arctern-dev
fi
