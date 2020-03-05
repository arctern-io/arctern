#!/bin/bash

conda install -y -q -n zgis_dev -c conda-forge -c arctern-dev libarctern arctern arctern-spark
conda clean --all -y