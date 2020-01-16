from distutils.core import setup
import os

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

setup(
    name = "zilliz_pyspark",
    py_modules = ['register'],
)

