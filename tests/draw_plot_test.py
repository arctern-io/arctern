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

# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
from pyspark.sql.types import *
from pyspark.sql import SparkSession

import matplotlib.pyplot as plt
# pylint: disable=c-extension-no-member
import cv2

from arctern_pyspark import register_funcs
from arctern_pyspark import plot

png_path = sys.path[0] + "/draw_map/"

def run_diff_png(baseline_png, compared_png, precision=0.0005):
    baseline_info = cv2.imread(baseline_png, cv2.IMREAD_UNCHANGED)
    compared_info = cv2.imread(compared_png, cv2.IMREAD_UNCHANGED)
    baseline_y, baseline_x = baseline_info.shape[0], baseline_info.shape[1]
    baseline_size = baseline_info.size

    compared_y, compared_x = compared_info.shape[0], compared_info.shape[1]
    compared_size = compared_info.size
    if compared_y != baseline_y or compared_x != baseline_x or compared_size != baseline_size:
        return False

    diff_point_num = 0
    for i in range(baseline_y):
        for j in range(baseline_x):
            baseline_rgba = baseline_info[i][j]
            compared_rgba = compared_info[i][j]

            baseline_rgba_len = len(baseline_rgba)
            compared_rgba_len = len(compared_rgba)
            if baseline_rgba_len != compared_rgba_len or baseline_rgba_len != 4:
                return False
            if compared_rgba[3] == baseline_rgba[3] and baseline_rgba[3] == 0:
                continue

            is_point_equal = True
            for k in range(3):
                tmp_diff = abs((int)(compared_rgba[k]) - (int)(baseline_rgba[k]))
                if tmp_diff > 1:
                    is_point_equal = False

            if not is_point_equal:
                diff_point_num += 1

    return ((float)(diff_point_num) / (float)(baseline_size)) <= precision


def run_test_plot(spark):
    register_funcs(spark)

    raw_data = []
    raw_data.extend([(0, 'polygon((0 0,0 1,1 1,1 0,0 0))')])
    raw_data.extend([(1, 'linestring(0 0,0 1,1 1,1 0,0 0)')])
    raw_data.extend([(2, 'point(2 2)')])

    wkt_collect = "GEOMETRYCOLLECTION(" \
                  "MULTIPOLYGON (((0 0,0 1,1 1,1 0,0 0)),((1 1,1 2,2 2,2 1,1 1)))," \
                  "POLYGON((3 3,3 4,4 4,4 3,3 3))," \
                  "LINESTRING(0 8,5 5,8 0)," \
                  "POINT(4 7)," \
                  "MULTILINESTRING ((1 1,1 2),(2 4,1 9,1 8))," \
                  "MULTIPOINT (6 8,5 7)" \
                  ")"
    raw_data.extend([(3, wkt_collect)])

    raw_schema = StructType([
        StructField('idx', LongType(), False),
        StructField('geo', StringType(), False)
    ])

    df = spark.createDataFrame(data=raw_data, schema=raw_schema)
    df.createOrReplaceTempView("geoms")
    df2 = spark.sql("select st_geomfromtext(geo) from geoms")

    # run baseline
    fig1, ax1 = plt.subplots()
    plot(ax1, df2)
    ax1.grid()
    baseline_png1 = png_path + "plot_test_1.png"
    fig1.savefig(baseline_png1)

    # run plot test
    fig2, ax2 = plt.subplots()
    plot(ax2, df2)
    ax2.grid()
    plot_test1 = png_path + "test_plot_test_1.png"
    fig2.savefig(plot_test1)

    spark.catalog.dropGlobalTempView("nyc_taxi")

    assert run_diff_png(baseline_png1, plot_test1)


if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Python Testmap") \
        .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    run_test_plot(spark_session)

    spark_session.stop()
