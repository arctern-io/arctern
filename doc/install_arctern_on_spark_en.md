# Installation Guide

This topic introduces how to install Arctern in Spark.

## TOC

## Prerequisites
<!-- TOC -->

- [TOC](#toc)
- [Prerequisites](#prerequisites)
- [Installing dependencies](#installing-dependencies)
- [Setting up Arctern Conda environment](#setting-up-arctern-conda-environment)
- [Installing Arctern](#installing-arctern)
- [Validating installation](#validating-installation)
- [Configure Python path for Spark](#configure-python-path-for-spark)
- [Test sample file](#test-sample-file)
- [Uninstalling Arctern](#uninstalling-arctern)
- [FAQ](#faq)
    - [Support for Spark](#support-for-spark)

<!-- /TOC -->
- CPU version

| Component     | Version              |
| -------- | ----------------- |
| Operating system | Ubuntu LTS 18.04  |
| Conda    | Miniconda Python3 |
| Spark    | 3.0               |

- GPU version

| Component          | Version              |
| ------------- | ----------------- |
| Operating system     | Ubuntu LTS 18.04  |
| Conda         | Miniconda Python3 |
| Spark         | 3.0               |
| CUDA          | 10.0              |
| Nvidia driver | 4.30              |

## Installing dependencies

- CPU version

  Use the following command to install dependencies for Arctern CPU version:

    ```bash
    $ sudo apt install libgl-dev libosmesa6-dev libglu1-mesa-dev
    ```

- GPU version

    Use the following command to install dependencies for Arctern GPU version:

    ```bash
    $ sudo apt install libgl1-mesa-dev libegl1-mesa-dev
    ```

## Setting up Arctern Conda environment

Use the following command to set up Arctern Conda environment:

```bash
$ conda create -n arctern
```

After the environment is created, you can use `conda env list` to check all Conda environments. The result should include the Arctern environment.

```bash
conda environments:
base         ...
arctern      ...
...
```

Enter Arctern environment:

```
$ conda activate arctern
```

**Note: The following steps must be performed in the Arctern environment**

## Installing Arctern

- CPU version

  Use the following command to install Arctern CPU version:

    ```shell
    $ conda install -y -q -n arctern -c conda-forge -c arctern-dev arctern-spark
    ```

- GPU version

  Use the following command to install Arctern CPU version:

    ```shell
    $ conda install -y -q -n arctern -c conda-forge -c arctern-dev/label/cuda10.0 libarctern
    $ conda install -y -q -n arctern -c conda-forge -c arctern-dev arctern arctern-spark
    ```

## Validating installation

In Python, import `arctern_gis` and `arctern_pyspark` to validate whether the installation is successful.

```python
Python 3.7.6 | packaged by conda-forge | (default, Jan 29 2020, 14:55:04)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import arctern_gis
>>> import arctern_pyspark
```

## Configure Python path for Spark

Add the following content to `conf/spark-env.sh`. `[path/to/your/conda]` is the installation path of Conda.

```bash
export PYSPARK_PYTHON=[path/to/your/conda]/envs/arctern/bin/python
export GDAL_DATA=[path/to/your/conda]/envs/arctern/share/gdal
export PROJ_LIB=[path/to/your/conda]/envs/arctern/share/proj
```

Check whether PySpark uses the Python path determined by `$PYSPARK_PYTHON`. `[path/to/your/spark]` is the installation path of Spark.

```python
[path/to/your/spark]/bin/pyspark
>>> import sys
>>> print(sys.prefix)
[path/to/your/conda]/envs/arctern
```

## Test sample file

Download test file.

```bash
wget https://raw.githubusercontent.com/zilliztech/arctern/conda/spark/pyspark/examples/gis/spark_udf_ex.py
```

Use the following command to submit Spark task. `[path/to/]spark_udf_ex.py` is the path of the test file.

```bash
# local mode
$ [path/to/your/spark]/bin/spark-submit [path/to/]spark_udf_ex.py

# standalone mode
$ [path/to/your/spark]/bin/spark-submit --master [spark service address] [path/to/]spark_udf_ex.py

# hadoop/yarn mode
$ [path/to/your/spark]/bin/spark-submit --master yarn [path/to/]spark_udf_ex.py
```

## Uninstalling Arctern

Use the following command to uninstall Arctern.

```shell
$ conda uninstall -n arctern libarctern arctern arctern-spark
```

## FAQ

### Support for Spark

Arctern can run in any mode of Spark. You must complete the following tasks for each host that runs Spark.

- Set up Conda environment
- Arctern Install Arctern
- Configure Spark environment variables

If Spark runs at `standalone` cluster mode, the Spark environment of the host that submit tasks must be consistent with the Spark environment of the cluster by:

- The absolute installation path of `spark` must be the same as each host in the cluster
- The absolute installation path of `conda` must be the same as each host in the cluster
- The name of the virtual environment must be the same as each host in the cluster
