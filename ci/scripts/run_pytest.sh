#!/bin/bash

set -e

cd "$(git rev-parse --show-toplevel)/python"

pytest
-v \
-x \
# --ignore "examples/example_test.py" \
# --ignore-glob "examples/*" \

