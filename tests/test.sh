rm -rf results
mkdir results
/usr/local/bin/spark/bin/spark-submit /home/liangliu/workspace/arctern/tests/spark_test.py

python collect_results.py
python compare.py
