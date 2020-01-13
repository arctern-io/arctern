from distutils.core import setup
from Cython.Build import cythonize

import os
import numpy as np
import pyarrow as pa



gis_modules = cythonize("zillizgis.pyx")

for ext in gis_modules:
    # The Numpy C headers are currently required
    ext.include_dirs.append(np.get_include())
    ext.include_dirs.append(pa.get_include())
    ext.include_dirs.append("../../../core/src/gis")
    ext.libraries.extend(pa.get_libraries())
    ext.library_dirs.extend(pa.get_library_dirs())

    if os.name == 'posix':
        ext.extra_compile_args.append('-std=c++11')

    # Try uncommenting the following line on Linux
    # if you get weird linker errors or runtime crashes
    ext.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))


setup(ext_modules=cythonize(gis_modules))
