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

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# Avoid gcc warnings
class BuildExt(build_ext):

    user_options = build_ext.user_options + [
        ('issymbol', None, "whether is symbol"),
    ]

    def initialize_options(self):
        super(BuildExt, self).initialize_options()
        self.issymbol = 0

    def gen_gis_core_modules(self):
        if self.issymbol:
            return self._gen_gis_symbol_modules()
        return self._gen_gis_core_modules()

    def _gen_gis_symbol_modules(self):
        #gis_core_modules = cythonize(Extension(name="arctern.arctern_core_",
        gis_core_modules = cythonize(Extension(name="arctern.arctern_core_",
                                               sources=["arctern/cython/arctern_core_symbol_.pyx"]))
        return gis_core_modules

    def _gen_gis_core_modules(self):
        import numpy as np
        import pyarrow as pa
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

    def finalize_options(self):
        self.distribution.ext_modules = self.gen_gis_core_modules()
        super(BuildExt, self).finalize_options()

    def build_extensions(self):
        # Avoid gcc warning "cc1plus: warning:command line option '-Wstrict-prototypes' is valid for C/ObjC but not for C++"
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        # Avoid gcc warning of unused-variable
        if '-Wno-unused-variable' in self.compiler.compiler_so:
            self.compiler.compiler_so.append('-Wno-unused-variable')
        super(BuildExt, self).build_extensions()

setup(
    cmdclass={'build_ext': BuildExt},
    packages=find_packages(),
)
