# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#from distutils.core import setup, Extension
import os
import numpy as np
import pyarrow as pa

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# Avoid gcc warnings
class BuildExt(build_ext):
    def build_extensions(self):
        # Avoid gcc warning "cc1plus: warning:command line option '-Wstrict-prototypes' is valid for C/ObjC but not for C++"
        self.compiler.compiler_so.remove('-Wstrict-prototypes')
        # Avoid gcc warning of unused-variable
        self.compiler.compiler_so.append('-Wno-unused-variable')
        super(BuildExt, self).build_extensions()


def gen_gis_core_modules():
    gis_core_modules = cythonize(Extension(name="arctern.arctern_core_",
                                           sources=["arctern/cython/arctern_core_.pyx"]))
    for ext in gis_core_modules:
        # The Numpy C headers are currently required
        ext.include_dirs.append(np.get_include())
        ext.include_dirs.append(pa.get_include())
        ext.libraries.extend(['arctern'] + pa.get_libraries())
        ext.library_dirs.extend(pa.get_library_dirs())

        if os.name == 'posix':
            ext.extra_compile_args.append('-std=c++11')

        # Try uncommenting the following line on Linux
        # if you get weird linker errors or runtime crashes
        #ext.define_macros.append(("_GLIBCXX_USE_CXX11_ABI", "0"))

    return gis_core_modules

setup(
    cmdclass={'build_ext': BuildExt},
    packages=find_packages(),
    ext_modules=gen_gis_core_modules(),
)
