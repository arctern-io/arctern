# Install Arctern Spark with Conda
本文介绍如何通过`Conda`环境安装`Arctern Spark`,并对各个模块进行测试。

## 安装前提

### 系统要求

| 操作系统    | 版本          |
| ---------- | ------------ |
| CentOS     | 7 或以上      |
| Ubuntu LTS | 16.04 或以上  |

### 软件要求

| 软件名称                    |
| -------------------------- |
| miniconda（推荐） 或者 anaconda     |

## 安装并配置conda

查看您当前conda环境配置，确认conda配置成功
```shell
$ conda env list

# conda environments:
#
base                  *  /opt/conda
...
```

如未成功配置Conda，请按照以下命令安装并配置Conda
```shell
# 安装conda
$ wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
$ /bin/bash ~/miniconda.sh -b -p /opt/conda
$ rm ~/miniconda.sh

# 配置conda
$ ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
$ echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
$ echo "conda activate base" >> ~/.bashrc
$ . /opt/conda/etc/profile.d/conda.sh
$ conda activate base
```

创建Conda Arctern环境
```shell
$ conda create -n arctern
```

激活Conda Arctern环境
```shell
$ conda activate arctern
```

## 安装`Arctern Spark`

CPU版本
```shell
conda install -y -q -n arctern -c conda-forge -c arctern-dev arctern-spark
```

CUDA 10.0版本
```shell
conda install -y -q -n arctern -c conda-forge -c arctern-dev/label/cuda10.0 libarctern
conda install -y -q -n arctern -c conda-forge -c arctern-dev arctern arctern-spark
```

## 测试`Arctern Spark`
```bash
conda activate arctern
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
export PYSPARK_PYTHON=/opt/conda/envs/arctern/bin/python
export GDAL_DATA=/opt/conda/envs/arctern/share/gdal
export PROJ_LIB=/opt/conda/envs/arctern/share/proj
```

### 检查pyspark是否使用$PYSPARK_PYTHON指定的python
```python
./bin/pyspark
>>> import sys
>>> print(sys.prefix)
/opt/conda/envs/arctern
```

### 下载pyspark测试用例
```bash
wget https://raw.githubusercontent.com/zilliztech/GIS/conda/spark/pyspark/examples/gis/spark_udf_ex.py
```

### 运行spark测试用例
```bash
./bin/spark-submit ./spark_udf_ex.py
```