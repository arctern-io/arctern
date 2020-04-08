import os
from pyspark.sql import SparkSession
make_path = "/home/liupeng/GIS/doc/api-doc"


def my_func():
    os.chdir(make_path)
    os.system("pwd")
    os.system("make clean")
    os.system("make html")

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("lp example") \
        .getOrCreate()

    my_func()
    spark.stop()
