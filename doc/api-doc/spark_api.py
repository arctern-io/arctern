import os
import threading

spark_cmd = "/home/liupeng/cpp/spark/spark-3.0.0-preview2/bin/spark-submit --master local /home/liupeng/GIS/spark/pyspark/arctern_pyspark/_wrapper_func.py"
make_clean = "make clean"
make_html = "make html"

def submit_spark_cmd():
    os.system(spark_cmd)

def submit_sphinx_cmd():
    os.system(make_clean)
    os.system(make_html)

if __name__=="__main__":
   p1 = threading.Thread(target=submit_spark_cmd)
   p2 = threading.Thread(target=submit_sphinx_cmd)
   p1.start()
  # p2.start()

