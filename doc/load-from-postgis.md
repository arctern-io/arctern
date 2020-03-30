# 从`postgis`加载数据
本文介绍`arctern`如何从`postgis`加载数据
- [使用文本文件](##使用文本文件)
- [使用JDBC](##使用JDBC)

---

## 使用文本文件
将`postgis`的数据另存为文本文件，然后交由`arctern`读取文本文件

假设`postgis`中有一张`polygon`表内容如下:
```sql
test=# select idx,st_astext(geos) from polygon;
 idx |                                                                 st_astext                                                                 
-----+-------------------------------------------------------------------------------------------------------------------------------------------
   0 | CIRCULARSTRING(0 2,-1 1,0 0,0.5 0,1 0,2 1,1 2,0.5 2,0 2)
   1 | COMPOUNDCURVE(CIRCULARSTRING(0 2,-1 1,0 0),(0 0,0.5 0,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2))
   2 | CIRCULARSTRING(0 2,-1 1,0 0,0.5 0,1 0,2 1,1 2,0.5 2,0 2)
   3 | CIRCULARSTRING(0 2,-1 1,0 2)
   4 | COMPOUNDCURVE(CIRCULARSTRING(0 2,-1 1,0 0),(0 0,0.5 0,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2))
   5 | COMPOUNDCURVE(CIRCULARSTRING(0 2,-1 1,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2))
   6 | COMPOUNDCURVE((0 2,-1 1,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2))
   7 | COMPOUNDCURVE(CIRCULARSTRING(0 0,1 1,1 0),(1 0,0 1))
   8 | COMPOUNDCURVE((0 0,1 1,1 0))
   9 | MULTICURVE((0 0,5 5),CIRCULARSTRING(4 0,4 4,8 4))
  10 | MULTICURVE((5 5,3 5,3 3,0 3),CIRCULARSTRING(0 0,0.2 1,0.5 1.4),COMPOUNDCURVE(CIRCULARSTRING(0 0,1 1,1 0),(1 0,0 1)))
  11 | MULTICURVE((5 5,3 5,3 3,0 3),CIRCULARSTRING(0 0,0.2 1,0.5 1.4),COMPOUNDCURVE((0 2,-1 1,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2)))
  12 | MULTICURVE((5 5,3 5,3 3,0 3),(0 0,0.2 1,0.5 1.4),COMPOUNDCURVE((0 2,-1 1,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2)))
  13 | CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0),(1 1,3 3,3 1,1 1))
  14 | CURVEPOLYGON(COMPOUNDCURVE(CIRCULARSTRING(0 0,2 0,2 1,2 3,4 3),(4 3,4 5,1 4,0 0)),CIRCULARSTRING(1.7 1,1.4 0.4,1.6 0.4,1.6 0.5,1.7 1))
  15 | MULTISURFACE(CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0),(1 1,3 3,3 1,1 1)))
  16 | MULTISURFACE(CURVEPOLYGON(CIRCULARSTRING(-2 0,-1 -1,0 0,1 -1,2 0,0 2,-2 0),(-1 0,0 0.5,1 0,0 1,-1 0)),((7 8,10 10,6 14,4 11,7 8)))
(17 rows)
```
使用这个语句把`polygon`表的数据另存为`csv`文件
```sql
\copy (select idx,st_astext(geos) as geos from polygon) to '/tmp/polygon.csv' delimiter ',' csv header;
```
`/tmp/polygon.csv`内容如下:
```csv
idx,geos
0,"CIRCULARSTRING(0 2,-1 1,0 0,0.5 0,1 0,2 1,1 2,0.5 2,0 2)"
1,"COMPOUNDCURVE(CIRCULARSTRING(0 2,-1 1,0 0),(0 0,0.5 0,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2))"
2,"CIRCULARSTRING(0 2,-1 1,0 0,0.5 0,1 0,2 1,1 2,0.5 2,0 2)"
3,"CIRCULARSTRING(0 2,-1 1,0 2)"
4,"COMPOUNDCURVE(CIRCULARSTRING(0 2,-1 1,0 0),(0 0,0.5 0,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2))"
5,"COMPOUNDCURVE(CIRCULARSTRING(0 2,-1 1,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2))"
6,"COMPOUNDCURVE((0 2,-1 1,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2))"
7,"COMPOUNDCURVE(CIRCULARSTRING(0 0,1 1,1 0),(1 0,0 1))"
8,"COMPOUNDCURVE((0 0,1 1,1 0))"
9,"MULTICURVE((0 0,5 5),CIRCULARSTRING(4 0,4 4,8 4))"
10,"MULTICURVE((5 5,3 5,3 3,0 3),CIRCULARSTRING(0 0,0.2 1,0.5 1.4),COMPOUNDCURVE(CIRCULARSTRING(0 0,1 1,1 0),(1 0,0 1)))"
11,"MULTICURVE((5 5,3 5,3 3,0 3),CIRCULARSTRING(0 0,0.2 1,0.5 1.4),COMPOUNDCURVE((0 2,-1 1,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2)))"
12,"MULTICURVE((5 5,3 5,3 3,0 3),(0 0,0.2 1,0.5 1.4),COMPOUNDCURVE((0 2,-1 1,1 0),CIRCULARSTRING(1 0,2 1,1 2),(1 2,0.5 2,0 2)))"
13,"CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0),(1 1,3 3,3 1,1 1))"
14,"CURVEPOLYGON(COMPOUNDCURVE(CIRCULARSTRING(0 0,2 0,2 1,2 3,4 3),(4 3,4 5,1 4,0 0)),CIRCULARSTRING(1.7 1,1.4 0.4,1.6 0.4,1.6 0.5,1.7 1))"
15,"MULTISURFACE(CURVEPOLYGON(CIRCULARSTRING(0 0,4 0,4 4,0 4,0 0),(1 1,3 3,3 1,1 1)))"
16,"MULTISURFACE(CURVEPOLYGON(CIRCULARSTRING(-2 0,-1 -1,0 0,1 -1,2 0,0 2,-2 0),(-1 0,0 0.5,1 0,0 1,-1 0)),((7 8,10 10,6 14,4 11,7 8)))"
```
`arctern`直接加载该`csv`文件，示例代码如下：
```python
from pyspark.sql import SparkSession
from arctern_pyspark import register_funcs
if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("polygon test") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    register_funcs(spark)
    spark.read.format("csv") \
         .option("header",True) \
         .option("delimiter",",") \
         .schema("idx long, geos string") \
         .load("/tmp/polygon.csv") \
         .createOrReplaceTempView("polygon")
    spark.sql("select idx, geos from polygon").show(20,0)
    spark.stop()
```

---

## 使用JDBC
假设`postgis`数据库信息如下：
|配置|值|
|---|---|
|ip address | 172.17.0.2 |
|port | 5432 |
|database name | test |
|user name | acterner |
|password | acterner |

使用如下命令测试能否连接`postgis`
```bash
psql test -h 172.17.0.2  -p 5432 -U arcterner
```
`arctern`使用`jdbc`加载`postgis`，`jdbc_postgis.py`示例代码如下:
```python
from pyspark.sql import SparkSession
from arctern_pyspark import register_funcs
if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("polygon test") \
        .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    register_funcs(spark)
    spark.read.format("jdbc") \
              .option("url", "jdbc:postgresql://172.17.0.2:5432/test?user=arcterner&password=arcterner") \
              .option("query", "select idx,st_astext(geos) as geos from polygon") \
              .load() \
              .createOrReplaceTempView("polygon")
    spark.sql("select idx, geos from polygon").show(20,0)
    spark.stop()
```
从[postgres官网](https://jdbc.postgresql.org/download.html)下载最新的`JDBC`驱动，这里下载的驱动为`postgresql-42.2.11.jar`，在提交`spark`任务时，需要指定`jdbc`驱动
```bash
./bin/spark-submit  --driver-class-path ~/postgresql-42.2.11.jar --jars ~/postgresql-42.2.11.jar ~/query_postgis.py 
```

---

## 参考
- [postgis](https://postgis.net)
- [postgres jdbc](https://jdbc.postgresql.org/download.html)
