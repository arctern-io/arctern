from distutils.core import setup, Extension
import pyarrow
import numpy
import os

MOD = "zillizgis"

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)

libs = ['zillizengine', 'gdal', 'proj'] + pyarrow.get_libraries()
lib_dir = ['/home/ljq/work/GIS/core/src/gis', '/home/ljq/czs_3rd/gis_build/third_party/lib']
lib_dirs = lib_dir + pyarrow.get_library_dirs()

extension = Extension(
    MOD,
    sources=[dir_path + '/cpp/zillizgis.cpp'],
    include_dirs=[numpy.get_include(), pyarrow.get_include()],
    libraries=libs,
    library_dirs=lib_dirs,
    extra_compile_args=["-std=c++11"],
    define_macros=[("_GLIBCXX_USE_CXX11_ABI", "0")]
)

setup(
        name=MOD,
        ext_modules=[extension]
)
