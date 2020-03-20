# conda环境搭建

## 安装前提

### 系统要求

| 操作系统    | 版本          |
| ---------- | ------------ |
| Ubuntu LTS | 18.04 或以上  |

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
base                  *  /home/xxx/miniconda3
...
```

如未成功配置Conda，请按照以下命令安装并配置Conda
```shell
# 安装conda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b
echo "source $HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc

安装过程中选择默认选项即可
安装完成之后，重启当前terminal
```

## 创建并使用arctern开发环境以及第三方库
1. 首先查看当前conda环境的所有环境名:  
`conda env list`  
如果之前有名称为arctern的conda环境，需要先移除  
`conda env remove -n arctern`  
2. 根据zgis.yml文件安装新的conda环境，名称为arctern  
`conda env create -f zgis.yml`  
运行之后安装了conda环境一些必要的库，但是还需要通过apt安装一些系统必要的库  
`sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev  libosmesa6-dev`
3. 安装好之后激活conda环境，并在此环境中进行下一步的代码编译  
`conda activate arctern`

# cpp代码编译以及单元测试的运行

#### 1. 克隆arctern项目至本地  
```
git clone https://github.com/zilliztech/arctern.git
```
注意：此后的所有工作均在conda中的arctern环境中运行。

#### 2. 切换到工程中的arctern/cpp目录然后运行下面的命令  

CPU Version
```
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DBUILD_UNITTEST=ON
make && make install
```

GPU Version
```
mkdir cpp/build && cd cpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} -DBUILD_UNITTEST=ON -DBUILD_WITH_GPU=ON
make && make install
```

cmake参数可选如下:
```
CMAKE_BUILD_TYPE     Release   [default]
BUILD_WITH_GPU       OFF       [default]
BUILD_UNITTEST       OFF       [default]
```

#### 3. 运行上述代码之后无错误，整个工程就编译成功了，然后运行下面的命令运行单元测试：  
```
./unittest/gis/gis_tests
./unittest/render/render_tests
```
运行完之后无错误输出就证明cpp编译和单元测试全部成功了。

# python封装以及单元测试的运行
上一步编译测试成功后
## Python包 arctern的编译和安装
- 进入conda环境
```
conda activate arctern
```
- 切换到工程中的arctern/python目录然后编译和安装arctern
```
rm build -rf
python setup.py build && python setup.py install
```



## 运行Python包 arctern的单元测试
到`arctern/python/test/geo`目录运行：
`py.test`
看到单元测试正确输出就代表结果正确！



# 在spark上运行

注意事项：spark请使用最新的spark-3.0.0-preview2.

## 编译arctern_pyspark包
- 切换到工程中的arctern/spark/pyspark目录然后编译和安装arctern

```
rm build -rf
python setup.py build && python setup.py install
```

## 确认是否安装成功  
在`python`命令行里输入`import arctern_pyspark`，看是否报错

## 设置链接选项和增加环境变量

修改`spark-defaults.conf`，只需要修改master节点配置即可，添加如下配置
```
spark.executorEnv.PROJ_LIB /home/xxx/miniconda3/envs/arctern_dev/share/proj
spark.executorEnv.GDAL_DATA /home/xxx/miniconda3/envs/arctern/share/gdal
```

## 修改环境变量  
在`spark-env.sh` 增加如下配置
```
export PYSPARK_PYTHON=/home/xxx/miniconda3/envs/arctern/bin/python
```

## 检查`pyspark`是否使用`$PYSPARK_PYTHON`指定的python
```
./bin/pyspark
>>> import sys
>>> print(sys.prefix)
/home/xxx/miniconda3/envs/arctern
```

## 运行单元测试
   
```
./bin/spark-submit /home/xxx/arctern/spark/pyspark/example/gis/spark_udf_ex.py
```
