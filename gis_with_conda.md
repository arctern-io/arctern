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



# spark udf封装以及单元测试的运行



# 和spark的整体使用


