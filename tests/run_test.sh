#!/bin/bash
/home/czp/workspace/spark/bin/spark-submit spark_test.py
python collect_results.py
python compare.py | grep FAILED
