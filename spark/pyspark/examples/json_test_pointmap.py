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

def run_curve_z(spark):
    curve_z_df = spark.read.json("/tmp/z_curve.json").cache()
    curve_z_df.createOrReplaceTempView("curve_z")
    register_funcs(spark)
    hex_data = spark.sql("select my_plot(x, y) from curve_z").collect()[0][0]
    str_hex_data = str(hex_data)
    import binascii
    binary_string = binascii.unhexlify(str_hex_data)
    with open('/tmp/hex_curve_z.png', 'wb') as png:
        png.write(binary_string)


if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python TestPointmap") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_curve_z(spark_session)

    spark_session.stop()
