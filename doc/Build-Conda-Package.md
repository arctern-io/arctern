# Build Conda Package
本文介绍如何为arctern生成`conda`包，并将生成的conda包上传到`anaconda cloud`

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
$ conda create -n arctern python=3.7
```

激活Conda Arctern环境
```shell
$ conda activate arctern
```

## 安装conda-build
```shell
$ conda install conda-build
```

## 构建`libarctern` Conda包

### CPU版本

安装包依赖
```shell
$ sudo apt install libgl-dev libosmesa6-dev libglu1-mesa-dev
```

构建Conda包
```shell
$ cd conda/recipes/libarctern/cpu
$ conda build . -c defaults -c conda-forge
```

出现如下信息表示成功构建`libarctern`包
```txt
+ exit 0

...

TEST END: /opt/conda/envs/arctern/conda-bld/linux-64/libarctern-0.0.0.dev-0.tar.bz2
```

### CUDA版本

安装包依赖
```shell
$ sudo apt install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev
```

构建Conda包
```shell
$ cd conda/recipes/libarctern/gpu
$ conda build . -c defaults -c conda-forge
```

出现如下信息表示成功构建`libarctern`包
```txt
+ exit 0

...

TEST END: /opt/conda/envs/arctern/conda-bld/linux-64/libarctern-0.0.0.dev-0.tar.bz2
```

## 构建`arctern` Conda包

构建Conda包
```shell
$ cd conda/recipes/arctern
$ conda build . -c defaults -c conda-forge
```

出现如下信息表示成功构建`arctern`包
```txt
+ exit 0

...

TEST END: /opt/conda/envs/arctern/conda-bld/linux-64/arctern-0.0.0.dev-0.tar.bz2
```

## 构建`arctern-spark` Conda包

构建Conda包
```shell
$ cd conda/recipes/arctern-spark
$ conda build . -c defaults -c conda-forge
```

出现如下信息表示成功构建`arctern-spark`包
```txt
+ exit 0

...

TEST END: /opt/conda/envs/arctern/conda-bld/linux-64/arctern-spark-0.0.0.dev-0.tar.bz2
```

## 上传Conda包到自己的Anaconda Cloud

### 注册`Anaconda Cloud`账号
anaconda : https://anaconda.org/

### 安装`Anaconda`客户端
```bash
conda install anaconda-client
```

### 上传Conda包

方式一:

登录`Anaconda`
```bash
anaconda login
```

上传Conda包，执行下面命令
```shell
$ export CONDA_FILE=`conda build ${RECIPES_PATH} -c conda-forge -c defaults --output`
$ LABEL_OPTION="--label ${LABELS}"
$ test -e ${CONDA_FILE}
$ anaconda upload ${LABEL_OPTION} --force ${CONDA_FILE}
```
`RECIPES_PATH`：对应Conda包的recipes路径，如：conda/recipes/libarctern/cpu
`LABELS`: Conda包的label，如：`main`(默认) 或者 `cuda10.0`等

方式二:

上传Conda包，执行下面命令
```shell
$ export CONDA_FILE=`conda build ${RECIPES_PATH} -c conda-forge -c defaults --output`
$ LABEL_OPTION="--label ${LABELS}"
$ test -e ${CONDA_FILE}
$ anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-arctern} ${LABEL_OPTION} --force ${CONDA_FILE}
```
`LABELS`: Conda包的label，如：`main`(默认) 或者 `cuda10.0`等
`RECIPES_PATH`：对应Conda包的recipes路径，如：conda/recipes/libarctern/cpu
`CONDA_USERNAME`: Anaconda Cloud的用户名，默认为 `arctern`
`MY_UPLOAD_KEY`: Anaconda Cloud的Token

获取Anaconda Cloud Token方式
  1.登陆`Anaconda Cloud`(https://anaconda.org/)
  2.点击右上角的用户名，跳转到`设置`
  3.在左侧面板，跳转到`访问`，然后要求输入密码
  4.现在，我们需要创建一个API Token。为其命名，并至少勾选`允许对API站点的读取访问权限`和`允许对API站点的写入权限`
  5.创建Token并复制保存
