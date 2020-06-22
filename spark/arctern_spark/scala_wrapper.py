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

import sys

if sys.version >= '3':
    basestring = str

from pyspark import SparkContext
from pyspark.sql.column import Column, _to_java_column, _create_column_from_name

from py4j.java_gateway import java_import

from pyspark.sql.types import UserDefinedType, StructField, BinaryType
from pyspark.sql import Row


class GeometryUDT(UserDefinedType):
    jvm = None

    @classmethod
    def sqlType(self):
        return StructField("wkb", BinaryType(), False)

    @classmethod
    def module(cls):
        return 'scala_wrapper'

    @classmethod
    def scalaUDT(cls):
        return 'org.apache.spark.sql.arctern.GeometryUDT'

    def serialize(self, obj):
        if obj is None:
            return None
        return Row(obj.toBytes)

    def deserialize(self, datum):
        binData = bytearray([x % 256 for x in datum])
        return binData


def import_scala_functions():
    sc = SparkContext._active_spark_context
    jvm = sc._jvm
    java_import(jvm, "org.apache.spark.sql.arctern.UdtRegistratorWrapper")
    jvm.UdtRegistratorWrapper.registerUDT()

    old_functions = jvm.functions
    java_import(jvm, "org.apache.spark.sql.arctern.functions")
    jvm.gis_functions = jvm.functions
    jvm.functions = old_functions


def _create_unary_function(name):
    def _(col, *args):
        sc = SparkContext._active_spark_context
        jc = getattr(sc._jvm.gis_functions, name)(_to_java_column(col), *args)
        return Column(jc)

    _.__name__ = name
    return _


def _create_binary_function(name, doc=""):
    def _(col1, col2):
        sc = SparkContext._active_spark_context
        if isinstance(col1, Column):
            arg1 = col1._jc
        elif isinstance(col1, basestring):
            arg1 = _create_column_from_name(col1)

        if isinstance(col2, Column):
            arg2 = col2._jc
        elif isinstance(col2, basestring):
            arg2 = _create_column_from_name(col2)

        jc = getattr(sc._jvm.gis_functions, name)(arg1, arg2)
        return Column(jc)

    _.__name__ = name
    _.__doc__ = doc
    return _


# functions that take one argument as input
_unary_functions = [
    "st_geomfromgeojson",
    "st_astext",
    "st_asgeojson",
    "st_centroid",
    "st_isvalid",
    "st_geometrytype",
    "st_issimple",
    "st_npoints",
    "st_envelope",
    "st_buffer",
    "st_precisionreduce",
    "st_simplifypreservetopology",
    "st_convexhull",
    "st_area",
    "st_length",
    "st_hausdorffdistance",
    "st_transform",
    "st_makevalid",
    "st_geomfromtext",
]

# functions that take two arguments as input
_binary_functions = [
    "st_point",
    "st_within",
    "st_intersection",
    "st_distance",
    "st_equals",
    "st_touches",
    "st_overlaps",
    "st_crosses",
    "st_contains",
    "st_intersects",
    "st_distancesphere",
]

import_scala_functions()

for _name in _unary_functions:
    globals()[_name] = _create_unary_function(_name)

for _name in _binary_functions:
    globals()[_name] = _create_binary_function(_name)

__all__ = [k for k, v in globals().items()
           if k in _unary_functions or k in _binary_functions and callable(v)]

__all__.sort()
