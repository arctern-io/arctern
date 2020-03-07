import os
from pyspark.sql import SparkSession

json_dir = '/home/liangliu/workspace/arctern/tests/data'
data_dir = '/home/liangliu/workspace/arctern/tests/data/csv'

def json2csv(spark):
    f_list = os.listdir(json_dir)
    for i in f_list:
        fs = os.path.splitext(i)
        if fs[1] == '.json':
            #print(i)
            # df = spark.read.json("/home/czp/tele/tests/data/%s" % i).cache()
            # df.write.csv("/home/czp/tele/tests/data/csv/%s.csv" % fs[0],sep='|')
            df = spark.read.json(os.path.join(json_dir, i)).cache()
            df.write.csv(os.path.join(data_dir, fs[0] + '.csv'), sep='|')



if __name__ == '__main__':
    # path = data_dir
    spark_session = SparkSession \
            .builder \
            .appName("json to csv script.") \
            .getOrCreate()

    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    json2csv(spark_session)
    spark_session.stop()

