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

import inspect

def _get_funcs_in_module(module):
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isfunction(obj):
            yield obj

def register_funcs(spark):
    """
    Register all functions provided by Arctern for the given SparkSession.

    :param spark: pyspark.sql.SparkSession, spark session instance.

    :example:
      >>> from pyspark.sql import SparkSession
      >>> from arctern_pyspark import register_funcs
      >>> spark_session = SparkSession.builder.appName("Python Arrow-in-Spark example").getOrCreate()
      >>> register_funcs(spark_session)
      >>>  test_data = []
      >>> test_data.extend([('POINT (10 10)',)])
      >>> buffer_df = spark_session.createDataFrame(data=test_data, schema=['geos']).cache()
      >>> buffer_df.createOrReplaceTempView("buffer")
      >>> spark_session.sql("select ST_AsText(ST_Transform(ST_GeomFromText(geos), 'epsg:4326', 'epsg:3857')) from buffer").show(100,0)
      +------------------------------------------------------------------------+
      |ST_AsText(ST_Transform(ST_GeomFromText(geos), 'epsg:4326', 'epsg:3857'))|
      +------------------------------------------------------------------------+
      |POINT (1113194.90793274 1118889.97485796)                               |
      +------------------------------------------------------------------------+
    """
    from . import  _wrapper_func
    all_funcs = _get_funcs_in_module(_wrapper_func)
    for obj in all_funcs:
        #print(obj.__name__, obj)
        spark.udf.register(obj.__name__, obj)
