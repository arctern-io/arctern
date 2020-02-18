#!/bin/bash

set -e

while getopts "e:h" arg
do
        case $arg in
             e)
                CONDA_ENV=$OPTARG # CONDA ENVIRONMENT
                ;;
             h) # help
                echo "

parameter:
-e: set conda activate environment
-h: help

usage:
./run_pytest.sh -e \${CONDA_ENV} [-h]
                "
                exit 0
                ;;
             ?)
                echo "ERROR! unknown argument"
        exit 1
        ;;
        esac
done

if [[ -n ${CONDA_ENV} ]]; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
fi

cd "$(git rev-parse --show-toplevel)/python"

pytest
-v \
-x \
# --ignore "examples/example_test.py" \
# --ignore-glob "examples/*" \
