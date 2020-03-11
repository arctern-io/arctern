# 安装说明
本文档介绍在 Spark 环境中安装 Arctern 的步骤。

## 大纲
* 安装要求
* 安装依赖库
* 创建 Conda 环境
* 安装 Arctern
* 验证是否安装成功
* 配置 Spark的 Python 路径
* 测试样例
* 卸载
* FAQ


## 安装要求

* CPU 版本

|  名称    |   版本     |
| ---------- | ------------ |
| 操作系统 |Ubuntu LTS 18.04|
| Conda  | Miniconda Python3  |
| Spark | 3.0  |


* GPU 版本

|  名称    |   版本     |
| ---------- | ------------ |
| 操作系统 |Ubuntu LTS 18.04|
| Conda | Miniconda Python3  |
| Spark | 3.0  |
|CUDA|10.0|
|Nvidia driver|4.30|



## 安装依赖库


* CPU 版本

  使用以下命令安装 Arctern CPU 版本的依赖库：
```bash
    sudo apt install libgl-dev libosmesa6-dev libglu1-mesa-dev
```

* GPU 版本


  使用以下命令安装 Arctern GPU 版本的依赖库：
```bash
    sudo apt install libgl1-mesa-dev libegl1-mesa-dev
```



## 创建 Arctern Conda 环境

### 创建 Arctern 虚拟环境

通过以下命令创建 Arctern Conda 环境：

`conda create -n arctern`

创建成功后，可以通过 `conda env list` 命令查看所有Conda环境，其输出结果应包含Arctern环境，类似如下：
  
  ```bash
  conda environments:
  base         ...
  arctern      ...
  ...
  ```

 进入 Arctern 环境：

  `conda activate arctern`


**注意：后续工作必须在 Arctern 环境中进行**



## 安装 Arctern


* CPU 版本

  执行以下命令在 Conda 环境中安装 Arctern CPU 版本：

```shell
    conda install -y -q -n arctern -c conda-forge -c arctern-dev arctern-spark
```

* GPU版本

  执行以下命令在 Conda 环境中安装 Arctern CPU 版本：  

```shell
    conda install -y -q -n arctern -c conda-forge -c arctern-dev/label/cuda10.0 libarctern
    conda install -y -q -n arctern -c conda-forge -c arctern-dev arctern arctern-spark
```



## 安装验证

进入 Python 环境，尝试导入 `arctern_gis` 和 `arctern_pyspark` 验证安装是否成功。

```python
Python 3.7.6 | packaged by conda-forge | (default, Jan 29 2020, 14:55:04)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import arctern_gis
>>> import arctern_pyspark
```

## 配置 Spark 的 Python 路径

在文件 `conf/spark-env.sh` 的最后添加以下内容。其中 `[path/to/your/conda]` 为Conda的安装路径。

```bash
export PYSPARK_PYTHON=[path/to/your/conda]/envs/arctern/bin/python
export GDAL_DATA=[path/to/your/conda]/envs/arctern/share/gdal
export PROJ_LIB=[path/to/your/conda]/envs/arctern/share/proj
```

通过如下方式，检查 PySpark 是否使用 $PYSPARK_PYTHON 指定的 Python 路径。其中 `[path/to/your/spark]` 为 Spark 的安装路径。

```python
[path/to/your/spark]/bin/pyspark
>>> import sys
>>> print(sys.prefix)
[path/to/your/conda]/envs/arctern
```



## 测试样例

下载测试文件

```bash
wget https://raw.githubusercontent.com/zilliztech/arctern/conda/spark/pyspark/examples/gis/spark_udf_ex.py
```

通过以下命令提交 Spark 任务，其中 `[path/to/]spark_udf_ex.py` 为测试文件所在的路径。

```bash
# local mode
[path/to/your/spark]/bin/spark-submit [path/to/]spark_udf_ex.py

# standalone mode
[path/to/your/spark]/bin/spark-submit --master [spark service address] [path/to/]spark_udf_ex.py

# hadoop/yarn mode
[path/to/your/spark]/bin/spark-submit --master yarn [path/to/]spark_udf_ex.py
```

## 卸载

在 Conda 环境中输入以下命令可卸载 Arctern

```shell
conda uninstall -n arctern libarctern arctern arctern-spark
```

## FAQ

### 对Spark的支持

Arctern 可以运行在 Spark 的各种模式下，需要在每台运行 Spark 的机器上，执行如下操作：

* 创建 Conda 虚拟环境
* 安装 Arctern
* 配置 Spark 环境变量

如果 Spark 运行在 `standalone` 集群模式下，提交任务机器的 Spark 环境需要与集群的 Spark 环境完全一致，包括以下几点：

* `spark` 安装的绝对路径与集群中每台机器完全一致
* `conda` 安装的绝对路径与集群中每个机器完全一致
* `conda` 虚拟环境名与集群中每个机器完全一致

