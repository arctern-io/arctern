# 安装说明
本文档介绍在 Conda 环境中通过构建本地 channel 安装 Arctern 的步骤，用于解决使用 Conda 在线安装 Arctern 时由于网络原因导致失败的问题。

## 大纲
* [安装要求](#prerequisities)
* [下载依赖项](#downloaddependencies)
* [为本地Conda Channel建立索引](#buildlocalchannelindex)
* [安装依赖库](#installdependency)
* [创建 Arctern Conda 环境](#constructenv)
* [从本地Conda channel安装 Arctern](#install)
* [验证是否安装成功](#verification)
* [配置 Spark的 Python 路径](#pathconfiguration)
* [测试样例](#test)
* [卸载](#uninstallation)
* [FAQ](#faq)


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



## <span id = "downloaddependencies">下载依赖项</span>

### 在线安装 Conda 环境依赖包

通过以下命令在线安装 conda-build 和 requests 包。

```bash
$ conda install conda-build
$ conda install requests
```

### 下载依赖项

下载以下文件并保存在同一路径下。

* [arctern_dependencies.txt](./scripts/arctern_dependencies.txt)

* [download_dependencies.py](./scripts/download_dependencies.py)

执行以下命令下载依赖项：

```bash
$ python download_dependencies.py
```

上述命令将下载 Arctern 所需的依赖项并保存在本地，并有类似如下输出：

```txt
Using environment list file: arctern_dependencies.txt
Getting  _libgcc_mutex-0.1-main.conda
URL: https://repo.anaconda.com/pkgs/main/linux-64/_libgcc_mutex-0.1-main.conda
	 Downloaded linux-64\_libgcc_mutex-0.1-main.conda
Getting  ca-certificates-2019.11.28-hecc5488_0.tar.bz2
URL: https://conda.anaconda.org/conda-forge/linux-64/ca-certificates-2019.11.28-hecc5488_0.tar.bz2
	 Downloaded linux-64\ca-certificates-2019.11.28-hecc5488_0.tar.bz2
Getting  ld_impl_linux-64-2.33.1-h53a641e_7.conda
URL: https://repo.anaconda.com/pkgs/main/linux-64/ld_impl_linux-64-2.33.1-h53a641e_7.conda
	 Downloaded linux-64\ld_impl_linux-64-2.33.1-h53a641e_7.conda
Getting  libgfortran-ng-7.3.0-hdf63c60_5.tar.bz2
URL: https://conda.anaconda.org/conda-forge/linux-64/libgfortran-ng-7.3.0-hdf63c60_5.tar.bz2
	 Downloaded linux-64\libgfortran-ng-7.3.0-hdf63c60_5.tar.bz2
Getting  libstdcxx-ng-9.1.0-hdf63c60_0.conda
URL: https://repo.anaconda.com/pkgs/main/linux-64/libstdcxx-ng-9.1.0-hdf63c60_0.conda
	 Downloaded linux-64\libstdcxx-ng-9.1.0-hdf63c60_0.conda
Getting  libwebp-base-1.1.0-2.tar.bz2
URL: https://conda.anaconda.org/conda-forge/linux-64/libwebp-base-1.1.0-2.tar.bz2
	 Downloaded linux-64\libwebp-base-1.1.0-2.tar.bz2

...
```

下载完成后，会在当前目录下创建具有如下结构的文件夹:

```txt
channel
├── label
│   └── cuda10.0
│       └── linux-64
│           └── ...
├── linux-64
│   └── ...
└── noarch
    └── ...
```



## <span id = "buildlocalchannelindex">为本地 Conda Channel 建立索引</span>

使用下述命令构建索引。Conda 将会在每个依赖包对应的目录下创建一个清单文件 `repodata.json`，从而将其作为本地channel。

```bash
$ cd channel
$ conda index .
```

运行命令会报告以下输出，其中[/path/to/channel]为channel目录所在路径。
```txt
updating index in: [/path/to/channel]/linux-64
...
updating index in: [/path/to/channel]/noarch
...
updating index in: [/path/to/channel]/label/cuda10.0/linux-64
...
```

完成此步骤后，channel 目录即可被 Conda 识别为本地 channel。



## <span id = "installdependency">安装依赖库</span>


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



## <span id = "constructenv">创建 Arctern Conda 环境</span>

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



## <span id = "install">从本地Conda channel安装 Arctern</span>


* CPU 版本

  执行以下命令在 Conda 环境中安装 CPU 版本 Arctern：

```shell
$ conda install -c file:///[path/to/channel] -n arctern arctern-spark --offline --override-channels
```

例如:

```shell
$ conda install -c file:///tmp/arctern-dependencies/channel -n arctern arctern-spark --offline --override-channels
```

* GPU版本

  执行以下命令在 Conda 环境中安装 GPU 版本 Arctern：  

```shell
    conda install -c file:///[path/to/channel]/label/cuda10.0 -n arctern libarctern --offline --override-channels
    conda install -c file:///[path/to/channel] -n arctern arctern arctern-spark --offline --override-channels
```

例如:

```shell
    conda install -c file:///tmp/arctern-dependencies/channel/label/cuda10.0 -n arctern libarctern --offline --override-channels
    conda install -c file:///tmp/arctern-dependencies/channel -n arctern arctern arctern-spark --offline --override-channels
```



## <span id = "verification">安装验证</span>

进入 Python 环境，尝试导入 `arctern` 和 `arctern_pyspark` 验证安装是否成功。

```python
Python 3.7.6 | packaged by conda-forge | (default, Jan 29 2020, 14:55:04)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import arctern
>>> import arctern_pyspark
```

## <span id = "pathconfiguration">配置 Spark 的 Python 路径</span>

在文件 `conf/spark-default.conf` 的最后添加以下内容。其中 `[path/to/your/conda]` 为Conda的安装路径。

```bash
spark.executorEnv.PROJ_LIB [path/to/your/conda]/envs/arctern/share/proj
spark.executorEnv.GDAL_DATA [path/to/your/conda]/envs/arctern/share/gdal
```

在文件 `conf/spark-env.sh` 的最后添加以下内容。其中 `[path/to/your/conda]` 为Conda的安装路径。

```bash
export PYSPARK_PYTHON=[path/to/your/conda]/envs/arctern/bin/python
```

通过如下方式，检查 PySpark 是否使用 $PYSPARK_PYTHON 指定的 Python 路径。其中 `[path/to/your/spark]` 为 Spark 的安装路径。

```python
[path/to/your/spark]/bin/pyspark
>>> import sys
>>> print(sys.prefix)
[path/to/your/conda]/envs/arctern
```



## <span id = "test">测试样例</span>

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

## <span id = "uninstallation">卸载</span>

在 Conda 环境中输入以下命令可卸载 Arctern

```shell
conda uninstall -n arctern libarctern arctern arctern-spark
```

## <span id = "faq">FAQ</span>

### 对Spark的支持

Arctern 可以运行在 Spark 的各种模式下，需要在每台运行 Spark 的机器上，执行如下操作：

* 创建 Conda 虚拟环境
* 安装 Arctern
* 配置 Spark 环境变量

如果 Spark 运行在 `standalone` 集群模式下，提交任务机器的 Spark 环境需要与集群的 Spark 环境完全一致，包括以下几点：

* `spark` 安装的绝对路径与集群中每台机器完全一致
* `conda` 安装的绝对路径与集群中每个机器完全一致
* `conda` 虚拟环境名与集群中每个机器完全一致