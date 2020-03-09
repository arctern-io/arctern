#!/bin/bash

rm build dist arctern_pyspark.egg-info -rf
python setup.py build
python setup.py install
