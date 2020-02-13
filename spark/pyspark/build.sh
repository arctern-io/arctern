#!/bin/bash

rm build dist zilliz_pyspark.egg-info -rf
python setup.py build
python setup.py install
