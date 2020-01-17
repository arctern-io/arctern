#from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


def gen_gis_core_modules():
    import os
    import numpy as np
    import pyarrow as pa
    gis_core_modules = cythonize(Extension(name="zilliz_gis.zilliz_gis_core_", sources=["zilliz_gis/cython/zilliz_gis_core_.pyx"]))

    for ext in gis_core_modules:
        # The Numpy C headers are currently required
        ext.include_dirs.append(np.get_include())
        ext.include_dirs.append(pa.get_include())
        ext.libraries.extend(['GIS'] + pa.get_libraries())
        ext.library_dirs.extend(pa.get_library_dirs())

        if os.name == 'posix':
            ext.extra_compile_args.append('-std=c++11')

        # Try uncommenting the following line on Linux
        # if you get weird linker errors or runtime crashes
        ext.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))

    return gis_core_modules

setup(
#    name = "zilliz_gis",
    packages=find_packages(),
    ext_modules=gen_gis_core_modules(),
)

