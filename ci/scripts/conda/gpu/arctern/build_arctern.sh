#!/usr/bin/env bash

set -e

CONDA_PYTHON=${CONDA_PYTHON:="3.7"}

if [ "$BUILD_ARCTERN" == '1' ]; then
  echo "Building arctern..."
  conda build --python "${CONDA_PYTHON}" conda/recipes/arctern/gpu -c defaults -c conda-forge -c nvidia
fi
