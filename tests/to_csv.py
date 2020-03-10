import os
from pyspark.sql import SparkSession
import sys

json_dir = sys.argv[1]
data_dir = sys.argv[2]

def json2csv(spark):
    f_list = os.listdir(json_dir)
    for i in f_list:
        fs = os.path.splitext(i)
        if fs[1] == '.json':
            df = spark.read.json(os.path.join(json_dir, i)).cache()
            df.write.csv(os.path.join(data_dir, fs[0] + '.csv'), sep='|')



if __name__ == '__main__':
    
    spark_session = SparkSession \
            .builder \
            .appName("json to csv script.") \
            .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    json2csv(spark_session)
    spark_session.stop()
    

