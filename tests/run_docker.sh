

docker run -d -v /home/liangliu/workspace/arctern:/home/arctern_test zilliz/arctern:0.2

docker exec -u root -it 
/usr/local/bin/spark/bin/spark-submit /home/liangliu/workspace/arctern/tests/spark_test.py

python collect_results.py
python compare.py
