# conda环境搭建
## 安装conda
conda环境的安装有很多教程，详见`https://www.jianshu.com/p/edaa744ea47d`。
## 创建并使用gis-dev开发环境以及第三方库
1. 首先查看当前conda环境的所有环境名:  
`conda env list`  
如果之前有名称为zgis_dev的conda环境，需要先移除  
`conda env remove -n zgis_dev`  
2. 根据zgis.yml文件安装新的conda环境，名称为zgis_dev  
`conda env create -f zgis.yml`  
运行之后安装了conda环境一些必要的库，但是还需要通过apt安装一些系统必要的库  
`sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev  libosmesa6-dev`
3. 安装好之后激活conda环境，并在此环境中进行下一步的代码编译  
`conda activate zgis_dev`

# cpp代码编译以及单元测试的运行
此部分所有工作均在conda中的zgis_dev环境中运行：
首先切换到工程中的GIS目录然后运行下面的命令：
```
mkdir build
cd build
cmake ../cpp -DCMAKE_INSTALL_PREFIX=$PWD/thirdparty -DBUILD_UNITTEST=ON
make
make install
```
运行上述代码之后无错误，整个工程就编译成功了，然后运行下面的命令运行单元测试：  
`./unittest/gis/cpp/geo_tests`  
运行完之后无错误输出就证明cpp编译和单元测试全部成功了。

# python封装以及单元测试的运行
上一步编译测试成功后
## Python包 zilliz_gis的编译和安装
- 修改 setup.cfg
    ```
    [build_ext]  
    Library-dirs= CMAKE_INSTALL_PREFIX/lib
    ```
其中CMAKE_INSTALL_PREFIX为cmake编译时指定的路径
- 运行python目录下的build.sh  
`./build.sh`
## 运行Python包 zilliz_gis的单元测试
1. 需要保证`LD_LIBRARY_PATH`中加入`CMAKE_INSTALL_PREFIX/lib`这个路径以及`cuda`的`lib`路径
2. 到`GIS/python/test/geo`目录运行：
`py.test`
看到单元测试正确输出就代表结果正确！



# 在spark上运行

注意事项：spark请使用最新的spark-3.0.0-preview2.

## 编译zilliz_pyspark包

```sh
cd GIS/spark/pyspark
./build.sh
```

## 确认是否安装成功  
在`python`命令行里输入`import zilliz_pyspark`，看是否报错

## 设置链接选项  

修改`spark-defaults.conf`，只需要修改master节点配置即可，添加如下配置
```
spark.driver.extraLibraryPath /home/xxx/gis/GIS/cpp/build/thirdparty/lib:/home/xxx/miniconda3/envs/zgis_dev/lib:/usr/local/cuda/lib64
spark.executor.extraLibraryPath /home/xxx/gis/GIS/cpp/build/thirdparty/lib:/home/xxx/miniconda3/envs/zgis_dev/lib:/usr/local/cuda/lib64
```

`/home/xxx/gis/GIS/cpp/build/thirdparty/lib`对应你编译cpp部分时，生成的库安装的地方

## 修改环境变量  
在`~/.bashrc`和`spark-env.sh` 增加如下配置
```
export PYSPARK_PYTHON=/home/xxx/miniconda3/envs/zgis_dev/bin/python
```
注1 : 如果当前用户默认`SHELL`不是`bash`，则在对应的rc文件中添加环境变量  
注2 : 设置环境变量后重启`SHELL`，确保当前`PYSPARK_PYTHON`生效

## 检查`pyspark`是否使用`$PYSPARK_PYTHON`指定的python
```
./bin/pyspark
>>> import sys
>>> print(sys.prefix)
/home/xxx/miniconda3/envs/gis2
```

## 运行单元测试
   
```
./bin/spark-submit /home/xxx/gis/GIS/spark/pyspark/example/gis/spark_udf_ex.py
```
