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
前提是 cpp 已经可以编译、测试跑通，假设cpp make 时指定的 安装目录是 CPP_INSTALL_DIR，根据上述设置，此时的CPP_INSTALL_DIR的相对路径为GIS/build/thirdparty。
## Python包 zilliz_gis的编译和安装
1. 修改 setup.cfg  
```
[build_ext]  
Library-dirs= CPP_INSTALL_DIR/lib
```
这里CPP_INSTALL_DIR代指编译安装cpp时指定的安装目录。  

2. 运行 build.sh  

注意：此时是在GIS文件夹下的Python目录中运行：  
`./build.sh`
## 运行Python包 zilliz_gis的单元测试  
测试之前，需要保证`LD_LIBRARY_PATH`中加入`CPP_INSTALL_DIR/lib`这个路径以及`cuda`的`lib`路径
然后切换到`GIS/python/test/geo`目录运行 :   
`py.test`  
看到单元测试正确输出就代表结果正确！
# spark udf封装以及单元测试的运行

# 和spark的整体使用
