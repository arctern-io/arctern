from distutils.core import setup, Extension
import os
import numpy as np
import pyarrow as pa

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

from distutils.core import setup
from Cython.Build import cythonize

gis_core_modules = cythonize("./cython/zilliz_gis_core.pyx", compiler_directives = {'language_level': 3})

for ext in gis_core_modules:
    # The Numpy C headers are currently required
    ext.include_dirs.append(np.get_include())
    ext.include_dirs.append(pa.get_include())
    ext.libraries.extend(['GIS'] + pa.get_libraries())
    # ext.library_dirs.extend(['/home/ljq/czs/GIS/cpp/build/core/lib'])

    if os.name == 'posix':
        ext.extra_compile_args.append('-std=c++11')

    # Try uncommenting the following line on Linux
    # if you get weird linker errors or runtime crashes
    ext.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))

setup(
    name = "zilliz_gis",
    ext_modules=cythonize(gis_core_modules),
)

