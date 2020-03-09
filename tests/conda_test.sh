#!/bin/bash

set -e

. /home/liangliu/anaconda3/etc/profile.d/conda.sh
conda env list
conda deactivate
conda activate zgis_dev

