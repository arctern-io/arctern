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


def run_st_point(spark):
    points_df = spark.read.json("/tmp/points.json").cache()
    points_df.createOrReplaceTempView("points")
    register_funcs(spark)
    spark.sql("select ST_Point_UDF(x, y) from points").show()

if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_st_point(spark_session)

    spark_session.stop()
