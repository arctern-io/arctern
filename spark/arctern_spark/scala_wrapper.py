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
# pylint: disable=protected-access

import sys
import databricks.koalas as ks
from py4j.java_gateway import java_import
from pyspark import SparkContext
from pyspark.sql.column import Column, _to_java_column
from pyspark.sql.types import UserDefinedType, StructField, BinaryType
from pyspark.sql import Row, DataFrame

ks.set_option('compute.ops_on_diff_frames', True)

if sys.version >= '3':
    basestring = str

class GeometryUDT(UserDefinedType):
    jvm = None

    @classmethod
    def sqlType(cls):
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
        binData = bytes([x % 256 for x in datum])
        return binData


def import_arctern_functions():
    sc = SparkContext._active_spark_context
    jvm = sc._jvm
    java_import(jvm, "org.apache.spark.sql.arctern")
    jvm.org.apache.spark.sql.arctern.UdtRegistratorWrapper.registerUDT()
    jvm.org.apache.spark.sql.arctern.UdfRegistrator.register(sc._jvm.SparkSession.getActiveSession().get())


def _create_function(name):
    def _(*args):
        sc = SparkContext._active_spark_context
        args = [_to_java_column(arg) for arg in args]
        jc = getattr(
            sc._jvm.org.apache.spark.sql.arctern.functions, name)(*args)
        return Column(jc)

    _.__name__ = name
    return _


def _creat_df_function(name):
    def _(*args):
        sc = SparkContext._active_spark_context
        sql_ctx = None
        for arg in args:
            if isinstance(arg, DataFrame):
                sql_ctx = arg.sql_ctx
                break
        args = [arg._jdf if isinstance(arg, DataFrame) else arg for arg in args]
        jdf = getattr(
            sc._jvm.org.apache.spark.sql.arctern.functions, name)(*args)
        return DataFrame(jdf, sql_ctx)

    _.__name__ = name
    return _


_functions = [
    "st_curvetoline",
    "st_geomfromgeojson",
    "st_astext",
    "st_aswkb",
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
    "st_transform",
    "st_makevalid",
    "st_geomfromtext",
    "st_union_aggr",
    "st_geomfromwkb",
    "st_envelope_aggr",
    "st_exteriorring",
    "st_isempty",
    "st_boundary",
    "st_scale",
    "st_affine",
    "st_translate",
    "st_rotate",
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
    "st_hausdorffdistance",
    "st_difference",
    "st_symdifference",
    "st_union",
    "st_polygonfromenvelope",
    "st_disjoint",
]

_df_functions = [
    "nearest_road",
    "near_road",
    "nearest_location_on_road",
]

import_arctern_functions()

for _name in _functions:
    globals()[_name] = _create_function(_name)
for _name in _df_functions:
    globals()[_name] = _creat_df_function(_name)

__all__ = [k for k, v in globals().items() if k in _functions + _df_functions and callable(v)]

__all__.sort()
