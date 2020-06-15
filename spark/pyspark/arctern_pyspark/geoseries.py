import pandas as pd
import arctern
import numpy as np
import arctern_pyspark

import os

from databricks.koalas import DataFrame
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.internal import _InternalFrame

os.environ['PYSPARK_PYTHON'] = "/home/zc/miniconda3/envs/koalas_env/bin/python"
os.environ['PYSPARK_DRIVER_PYTHON'] = "/home/zc/miniconda3/envs/koalas_env/bin/python"

import databricks.koalas as ks
from pyspark.sql.functions import col, lit
from pyspark.sql import SparkSession
from pyspark.sql.types import *

ks.set_option('compute.ops_on_diff_frames', True)


class GeoSeries(ks.Series):

    @property
    def area(self):

        from arctern_pyspark import _wrapper_func
        from pyspark.sql.functions import col, lit

        _kdf = self.to_dataframe()
        sdf = _kdf.to_spark()
        ret = sdf.select(_wrapper_func.ST_Area(_wrapper_func.ST_GeomFromText(col("col1"))).alias("col1"))  # spark dataframe
        kdf = ret.to_koalas()
        from databricks.koalas.internal import _InternalFrame
        internal = _InternalFrame(
            kdf._internal.spark_frame,
            index_map=_kdf._internal.index_map,
            column_labels=_kdf._internal.column_labels,
            column_label_names=_kdf._internal.column_label_names,
        )

        return ks.Series(internal, anchor=kdf)

    def equals(self, other):

        from arctern_pyspark import _wrapper_func
        from pyspark.sql.functions import col, lit
        _kdf = ks.DataFrame(data=self)
        _kdf["col2"] = other
        print(_kdf)
        sdf = _kdf.to_spark()

        ret = sdf.select(_wrapper_func.ST_Equals(_wrapper_func.ST_GeomFromText(col("col1")),
                                                 _wrapper_func.ST_GeomFromText(col("col2"))).alias("xixi"))  # spark dataframe

        kdf = ret.to_koalas()

        from databricks.koalas.internal import _InternalFrame
        internal = _InternalFrame(
            kdf._internal.spark_frame,
            index_map=kdf._internal.index_map,
            column_labels=kdf._internal.column_labels,
            column_label_names=kdf._internal.column_label_names,
        )

        return ks.Series(internal, anchor=kdf)


def test_create_empty_df():

    spark = SparkSession \
        .builder \
        .appName("Python Arrow-in-Spark example") \
        .getOrCreate()

    schema = StructType([
        StructField("c", IntegerType(), True)])

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    print("haha")
    df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema=schema)
    print(df)


rows = 1
data_series = [
                  'POLYGON ((1 1,1 2,2 2,2 1,1 1))',
                  'POLYGON ((0 0,0 4,2 2,4 4,4 0,0 0))',
                  'POLYGON ((0 0,0 4,4 4,0 0))',
              ] * rows

countries = ['London', 'New York', 'Helsinki'] * rows
s1 = GeoSeries(data_series, name="col1", index=countries)
s2 = GeoSeries(data_series, name="col2", index=countries)

ret2 = s1.area
ret = s1.equals(s2)
print(ret)
print(ret2)
# print(ret)
# t = s.to_pandas()
# print(t)