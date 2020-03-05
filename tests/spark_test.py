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
from zilliz_pyspark import register_funcs
import random
import shutil
import os
import json


base_dir = './data/'

def clear_result_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except IOError:
        # print('IO error happened while deal with dir: %s' % dir_name)   
        pass
        

def get_data(base_dir, fname):
    return os.path.join(base_dir, fname)

def to_json_file(file_path, content):
    with open(file_path, 'w') as f:
        json.dump(content, f)

def to_txt(file_dir, df):
    df.write.text(file_dir)

def to_json(file_dir, df):
    df.write.json(file_dir)

def get_test_config(config_file):
    with open(config_file, 'r') as f:
        configs = json.load(f)
    return configs

# ------------------------------------ test part ------------------------------------
def run_test_st_point(spark):
    # ***********************************************
    # generate st_point_udf from json data loaded
    # ***********************************************

    data = "points.json"
    table_name = 'test_points'
    sql = "select st_point_udf(x, y) from test_points"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_envelope_aggr_1(spark):
    
    data = "envelope_aggr.json"
    table_name = 'test_envelope_aggr_1'
    sql = "select st_envelope_aggr_udf(geos) as my_envelope from test_envelope_aggr_1"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_envelope_aggr_2(spark):
    
    data = "points.json"
    table_name = 'envelope_aggr_2'
    sql = "select st_envelope_aggr_udf(arealandmark) from (select st_point_udf(x, y) as arealandmark from envelope_aggr_2)"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)
    
def run_test_st_isvalid_1(spark):
    
    data = "isvalid.json"
    table_name = 'test_isvalid'
    sql = "select st_isvalid_udf(geos) as is_valid from test_isvalid"

    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)
    

# def run_test_st_isvalid_2(spark):
    
#     data = "points.json"
#     table_name = 'points_df'
#     sql = "select st_isvalid_udf(null)"

#     try:
#         rs = spark.sql(sql).cache()
#     except Exception as e:
#         print(str(e))

def run_test_union_aggr_1(spark):
    data = 'polygonfromenvelope.json'
    table_name = 'test_union_aggr_1'
    sql = "select st_union_aggr_udf(myshape) as union_aggr from (select st_polygonfromenvelope_udf(a,c,b,d) as myshape from test_union_aggr_1)"

    df = spark.read.json(get_data(base_dir, data)).cache()
    df.show()
    df.printSchema()
    df.createOrReplaceTempView(table_name)

    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_union_aggr_2(spark):
    data = 'union_aggr.json'
    table_name = 'test_union_aggr_2'
    sql = "select st_union_aggr_udf(geos) as union_aggr from test_union_aggr_2"

    df = spark.read.json(get_data(base_dir, data)).cache()
    df.show()
    df.printSchema()
    df.createOrReplaceTempView(table_name)

    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_intersection(spark):

    data = "intersection.json"
    table_name = 'test_intersection'
    sql = "select st_intersection_udf(left, right) from test_intersection"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_convexhull(spark):
    data = "convexhull.json"
    table_name = 'test_convexhull'
    sql = "select ST_convexhull_UDF(geos) as geos from test_convexhull"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_buffer(spark):
    data = "buffer.json"
    table_name = 'test_buffer'
    sql = "select st_buffer_udf(geos, 1) as geos from test_buffer"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_envelope(spark):
    data = "envelope.json"
    table_name = 'test_envelope'
    sql = "select st_envelope_udf(geos) as geos from test_envelope"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_centroid(spark):
    data = "centroid.json"
    table_name = 'test_centroid'
    sql = "select st_centroid_udf(geos) as my_centroid from test_centroid"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_length(spark):
    data = "length.json"
    table_name = 'test_length'
    sql = "select st_length_udf(geos) as my_length from test_length"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_area(spark):
    data = "area.json"
    table_name = 'test_area'
    sql = "select st_area_udf(geos) as my_area from test_area"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_distance(spark):
    data = "distance.json"
    table_name = 'test_distance'
    sql = "select st_distance_udf(left, right) as my_distance from test_distance"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_issimple(spark):
    data = "issimple.json"
    table_name = 'test_issimple'
    sql = "select st_issimple_udf(geos) from test_issimple"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_npoints(spark):
    data = "npoints.json"
    table_name = 'test_npoints'
    sql = "select st_npoints_udf(geos) as my_npoints from test_npoints"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_geometrytype(spark):
    data = "geom.json"
    table_name = 'test_gt'
    sql = "select st_geometrytype_udf(geos) as geos from test_gt"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()

    to_json("results/%s" % table_name, rs)

def run_test_st_transform(spark):
    data = "transform.json"
    table_name = 'test_transform'
    sql = "select st_transform_udf(geos, 'epsg:4326', 'epsg:3857') as geos from test_transform"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    
    to_json("results/%s" % table_name, rs)

def run_test_st_intersects(spark):
    
    data = "intersects.json"
    table_name = 'test_intersects'
    sql = "select st_intersects_udf(left, right) as geos from test_intersects"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_contains(spark):
    
    data = "contains.json"
    table_name = 'test_contains'
    sql = "select st_contains_udf(left, right) as geos from test_contains"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_within(spark):
    
    data = "within.json"
    table_name = 'test_within'
    sql = "select st_within_udf(left, right) as geos from test_within"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_equals_1(spark):
    
    data = "equals.json"
    table_name = 'test_equals'
    sql = "select st_equals_udf(left, right) as geos from test_equals"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_equals_2(spark):
    
    data = "equals2.json"
    table_name = 'test_equals_2'
    sql = "select st_equals_udf(st_envelope_udf(left), right) as geos from test_equals_2"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_crosses(spark):
    
    data = "crosses.json"
    table_name = 'test_crosses'
    sql = "select st_crosses_udf(left, right) as geos from test_crosses"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_overlaps(spark):
    
    data = "overlaps.json"
    table_name = 'test_overlaps'
    sql = "select st_overlaps_udf(left, right) as geos from test_overlaps"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_touches(spark):
    
    data = "touches.json"
    table_name = 'test_touches'
    sql = "select st_touches_udf(left, right) as geos from test_touches"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_makevalid(spark):
    data = "makevalid.json"
    table_name = 'test_makevalid'
    sql = "select st_makevalid_udf(geos) as geos from test_makevalid"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_polygonfromenvelope(spark):
    data = "polygonfromenvelope.json"
    table_name = 'test_polygonfromenvelope'
    sql = "select st_polygonfromenvelope_udf(a, c, b, d) as geos from test_polygonfromenvelope"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

def run_test_st_simplifypreservetopology(spark):
    data = "simplifypreservetopology.json"
    table_name = 'test_simplifypreservetopology'
    sql = "select st_simplifypreservetopology_udf(geos, 10) as geos from test_simplifypreservetopology"
    
    df = spark.read.json(get_data(base_dir, data)).cache()
    df.printSchema()
    df.show()
    df.createOrReplaceTempView(table_name)
    
    rs = spark.sql(sql).cache()
    rs.printSchema()
    rs.show()
    to_json("results/%s" % table_name, rs)

# def run_test(spark, config):
#     print('================= Run from config =================')
#     data = config['data']
#     table_name = config['table_name']
#     sql = config['sql']
#     print(data)
#     print(table_name)
#     print(sql)
#     df = spark.read.json(get_data(base_dir, data)).cache()
#     df.show()
#     df.createOrReplaceTempView(table_name)
#     rs = spark.sql(sql).cache()
#     # to_txt("results/%s" % table_name, df)

import inspect
import sys

if __name__ == "__main__":

    # url = '192.168.1.65'
    url = 'local'
    spark_session = SparkSession.builder.appName("Python zgis sample").master(url).getOrCreate()
    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    clear_result_dir('results')
    register_funcs(spark_session)
    
    # confs = get_test_config('test.json')
    # for c in confs:
    #     run_test(spark_session, c)

    run_test_st_point(spark_session)
    run_test_envelope_aggr_1(spark_session)
    run_test_envelope_aggr_2(spark_session)
    run_test_union_aggr_2(spark_session)
    run_test_st_isvalid_1(spark_session)
    run_test_st_intersection(spark_session)
    run_test_st_convexhull(spark_session)
    run_test_st_buffer(spark_session)
    run_test_st_envelope(spark_session)
    run_test_st_centroid(spark_session)
    run_test_st_length(spark_session)
    run_test_st_area(spark_session)
    run_test_st_distance(spark_session)
    run_test_st_issimple(spark_session)
    run_test_st_npoints(spark_session)
    run_test_st_geometrytype(spark_session)
    run_test_st_transform(spark_session)
    run_test_st_intersects(spark_session)
    run_test_st_contains(spark_session)
    run_test_st_within(spark_session)
    run_test_st_equals_1(spark_session)
    run_test_st_equals_2(spark_session)
    run_test_st_crosses(spark_session)
    run_test_st_overlaps(spark_session)
    run_test_st_touches(spark_session)
    run_test_st_makevalid(spark_session)
    run_test_st_polygonfromenvelope(spark_session)
    run_test_st_simplifypreservetopology(spark_session)


    spark_session.stop()

    
