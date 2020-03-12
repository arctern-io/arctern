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

from pyspark.sql import SparkSession
from arctern_pyspark import register_funcs


def intersection_gen():
    with open('/tmp/intersection.json', 'w') as file:
        file.write('{"left": "POINT(0 0)", "right": "LINESTRING ( 2 0, 0 2  )"}\n')
        file.write('{"left": "POINT(0 0)", "right": "LINESTRING ( 0 0, 0 2  )"}\n')


def run_st_intersection(spark):
    test_df = spark.read.json("/tmp/intersection.json").cache()
    test_df.createOrReplaceTempView("intersection")
    register_funcs(spark)
    spark.sql("select ST_Intersection(left, right) from intersection").show()


if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    intersection_gen()

    run_st_intersection(spark_session)

    spark_session.stop()
