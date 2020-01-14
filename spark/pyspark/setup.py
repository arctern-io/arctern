from distutils.core import setup, Extension
import os
import numpy as np
import pyarrow as pa

# MOD = "zilliz_gis"

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

from distutils.core import setup

setup(
    name = "zilliz_pyspark",
    py_modules = ['register'],
)

