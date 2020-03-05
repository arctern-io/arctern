## Install arctern with conda
### 环境
- 操作系统 : ubuntu 16.04 64位
- Miniconda

### 安装依赖包
- sudo apt install libgl-dev libosmesa6-dev libglu1-mesa-dev

### 安装Miniconda
Miniconda官网 : https://docs.conda.io/en/latest/miniconda.html

### 搭建Arctern环境
```bash
conda create -n arctern_dev -c nezha2017 -c conda-forge arctern
```

### 启动`arctern_dev`并测试
```bash
conda activate arctern_dev
python
```
导入`zilliz_gis`,`zilliz_pyspark`验证安装成功
```python
Python 3.8.1 | packaged by conda-forge | (default, Jan 29 2020, 14:55:04) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import zilliz_gis
>>> import zilliz_pyspark
```

### 安装`java8`运行环境
```bash
apt install openjdk-8-jre
```

### 下载最新版本的spark并解压
Apache Spark : https://spark.apache.org/downloads.html

### 配置spark的python路径
在文件`conf/spark-env.sh `添加以下内容，其中`/opt/conda`由`Miniconda`具体安装位置决定
```bash
export PYSPARK_PYTHON=/opt/conda/envs/arctern_dev/bin/python
export GDAL_DATA=/opt/conda/envs/arctern_dev/share/gdal
export PROJ_LIB=/opt/conda/envs/arctern_dev/share/proj
```

### 检查pyspark是否使用$PYSPARK_PYTHON指定的python
```python
./bin/pyspark
>>> import sys
>>> print(sys.prefix)
/opt/conda/envs/arctern_dev
```

### 下载pyspark测试用例
```bash
wget https://raw.githubusercontent.com/zilliztech/arctern/conda/spark/pyspark/examples/gis/spark_udf_ex.py
```

### 运行spark测试用例
```bash
./bin/spark-submit ./spark_udf_ex.py
```
