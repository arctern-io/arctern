# Installation with local Conda channel

This document introduces how to install Arctern with local built Conda channel. This avoids installation failures caused by network issues.

## TOC

<!-- TOC -->

- [Prerequisites](#prerequisites)
- [Installing Arctern](#installing-arctern)
    - [Install conda-build and requests online](#install-conda-build-and-requests-online)
    - [Download dependencies](#download-dependencies)
- [Building indexes for local Conda Channel](#building-indexes-for-local-conda-channel)
- [Installing dependencies](#installing-dependencies)
- [Setting up Arctern Conda environment](#setting-up-arctern-conda-environment)
- [Installing Arctern from Conda channel](#installing-arctern-from-conda-channel)
- [Validating installation](#validating-installation)
- [Configuring Python path for Spark](#configuring-python-path-for-spark)
- [Test case](#test-case)
- [Uninstalling Arctern](#uninstalling-arctern)
- [FAQ](#faq)
    - [Support for Spark](#support-for-spark)

<!-- /TOC -->

## Prerequisites

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


## Installing Arctern

### Install conda-build and requests online

```bash
$ conda install conda-build
$ conda install requests
```

### Download dependencies

Download the following files and save them to the same folder.

* [arctern_dependencies.txt](./scripts/arctern_dependencies.txt)

* [download_dependencies.py](./scripts/download_dependencies.py)

Run the following command to download dependencies:

```bash
$ python download_dependencies.py
```

The command downloads required dependencies needed by Arctern and saves to local disk. The script has the following output:

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

```

After download, a folder with the following structure is created:

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

## Building indexes for local Conda Channel

Please use the following command to build indexes. Conda creates a file `repodata.json` in the folder corresponding to each dependency to make it a local channel.

```bash
$ cd channel
$ conda index .
```

Run the command to return the following output. [/path/to/channel] is the directory to the channel folder.

```txt
updating index in: [/path/to/channel]/linux-64
...
updating index in: [/path/to/channel]/noarch
...
updating index in: [/path/to/channel]/label/cuda10.0/linux-64
...
```

Then, the root folder (channel) can be recognized as local channel.


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

## Installing Arctern from Conda channel


* CPU version

  Use the following command to install Arctern CPU version:

```shell
$ conda install -c file:///[path/to/channel] -n arctern arctern-spark --offline --override-channels
```

例如:

```shell
$ conda install -c file:///tmp/arctern-dependencies/channel -n arctern arctern-spark --offline --override-channels
```

* GPU version

  Use the following command to install Arctern GPU version:  

```shell
    conda install -c file:///[path/to/channel]/label/cuda10.0 -n arctern libarctern --offline --override-channels
    conda install -c file:///[path/to/channel] -n arctern arctern arctern-spark --offline --override-channels
```

For example:

```shell
    conda install -c file:///tmp/arctern-dependencies/channel/label/cuda10.0 -n arctern libarctern --offline --override-channels
    conda install -c file:///tmp/arctern-dependencies/channel -n arctern arctern arctern-spark --offline --override-channels
```

## Validating installation

In Python, import `arctern` and `arctern_pyspark` to validate whether the installation is successful.

```python
Python 3.7.6 | packaged by conda-forge | (default, Jan 29 2020, 14:55:04)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import arctern
>>> import arctern_pyspark
```

## Configuring Python path for Spark

Add the following content to `conf/spark-default.conf`. `[path/to/your/conda]` is the installation path of Conda.

```bash
spark.executorEnv.PROJ_LIB [path/to/your/conda]/envs/arctern/share/proj
spark.executorEnv.GDAL_DATA [path/to/your/conda]/envs/arctern/share/gdal
```

Add the following content to `conf/spark-env.sh`. `[path/to/your/conda]` is the installation path of Conda.

```bash
export PYSPARK_PYTHON=[path/to/your/conda]/envs/arctern/bin/python
```

Check whether PySpark uses the Python path determined by `$PYSPARK_PYTHON`. `[path/to/your/spark]` is the installation path of Spark.

```python
[path/to/your/spark]/bin/pyspark
>>> import sys
>>> print(sys.prefix)
[path/to/your/conda]/envs/arctern
```

## Test case

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