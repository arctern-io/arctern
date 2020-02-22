## Build Conda Package
本文介绍如何编译`cpu`版的`arctern`，生成`conda`包，并上传到`anaconda cloud`

### 环境
- 操作系统 : ubuntu 16.04 64位
- Miniconda

### 安装依赖包
```bash
sudo apt install libgl-dev libosmesa6-dev libglu1-mesa-dev
```
需要安装`cmake`和`gcc`编译环境

### 安装Miniconda
Miniconda官网 : https://docs.conda.io/en/latest/miniconda.html

### 编辑`zgis.yml`
```yml
name: zgis_dev
dependencies:
- conda-forge::gdal=3.0.4
- conda-forge::pyarrow
- conda-forge::gtest
- conda-forge::rapidjson
- conda-forge::Cython
- conda-forge::pytest
- conda-forge::pylint
- conda-forge::conda-build
```

### 新建conda环境`zgis_dev`
```bash
conda env create -f zgis.yml
```

### 激活`zgis_dev`
```bash
conda activate zgis_dev
```

### 下载`arctern`源码
```bash
git clone https://github.com/zilliztech/GIS.git
```

### 新建`recipes`目录
```bash
mkdir -p GIS/conda/recipes/cpu
cd GIS/conda/recipes/cpu
```

### 新建`meta.yaml`
`meta.yaml`文件位于目录`GIS/conda/recipes/cpu`，内如如下:

```yaml
package:
  name: arctern
  version: 0.1.0

source:
  path: ../../..

build:
  number: 0

requirements:
  build:
    - python ==3.7.6
    - gdal ==3.0.4
    - pyarrow ==0.16.0
    - gtest
    - rapidjson ==1.1.0
    - Cython ==0.29.15
    - pytest
  run:
    - python ===3.7.6
    - gdal ==3.0.4
    - pyarrow ===0.16.0
    - rapidjson ==1.1.0
    - Cython ==0.29.15

test:
  commands:
    - python -c "import zilliz_gis"
    - python -c "import zilliz_pyspark"

about:
  home: http://www.github.com/zilliztech/GIS
  license: Apache-2.0
  license_family: Apache
  license_file: LICENSE
  summary:  arctern core library

```
<font color=red>注意事项</font><br/>
1. `gdal`和`pyarrow`的版本必须与当前conda环境中的环境中的版本一致,不一致可能出现编译失败的情况
2. `conda list` 命令可以查看当前conda环境中所有软件的版本
3. `source/path`为`meta.yaml`到项目根目录的相对路径

### 新建`build.sh`
`build.sh`文件位于目录`GIS/conda/recipes/cpu`，内如如下:
```bash
#!/bin/bash
if [ -d cpp/build ];then
    rm -rf cpp/build
fi
mkdir cpp/build
pushd cpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBUILD_UNITTEST=OFF  -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF
make install
popd

pushd python
python setup.py build
python setup.py install
popd

pushd spark/pyspark
python setup.py build
python setup.py install
popd

```

### 构建`arctern`包
```
conda build -c conda-forge .
```
出现如下信息表示成功构建`arctern`包
```txt
# Automatic uploading is disabled
# If you want to upload package(s) to anaconda.org later, type:

anaconda upload /opt/conda/envs/zgis_dev/conda-bld/linux-64/arctern-0.1.0-py37_0.tar.bz2

# To have conda build upload to anaconda.org automatically, use
# $ conda config --set anaconda_upload yes

anaconda_upload is not set.  Not uploading wheels: []
####################################################################################
Resource usage summary:

Total time: 0:04:48.4
CPU usage: sys=0:00:00.7, user=0:00:10.1
Maximum memory usage observed: 566.3M
Total disk usage observed (not including envs): 26.9K

```

### 注册`anaconda cloud`账号
anaconda : https://anaconda.org/

### 安装`anaconda`客户端
```bash
conda install anaconda-client
```

### 登录`anaconda`
```bash
anaconda login
```

### 上传`arctern`包
```bash
anaconda upload /opt/conda/envs/zgis_dev/conda-bld/linux-64/arctern-0.1.0-py37_0.tar.bz2
```
如果包已经存在,则添加`--force`选项覆盖
```bash
anaconda upload --force /opt/conda/envs/zgis_dev/conda-bld/linux-64/arctern-0.1.0-py37_0.tar.bz2
```





