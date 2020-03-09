#!/bin/bash
/opt/spark/bin/spark-submit /home/arctern_test/tests/spark_test.py

python collect_results.py
python compare.py
