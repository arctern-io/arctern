# docker compose部署

## 安装前提

### 系统要求


| 操作系统    | 版本          |
| ---------- | ------------ |
| CentOS     | 7 或以上      |
| Ubuntu LTS | 16.04 或以上  |

### 软件要求


| 软件名称        | 版本          | 备注 |
| ----------     | ------------ |      |
| Docker         | 17.06.0 或以上|      |
| Docker compose | 1.17.1 或以上 |      |
| Nvidia Docker  | Version 2    | 可选  |

## 配置Docker

### 确认Docker状态
在您的宿主机上[安装 Docker](https://docs.docker.com/install/)

确认 Docker daemon 正在运行：

```shell
$ docker info
```

如果无法正常打印 Docker 相关信息，请启动 **Docker** daemon.

> 提示：在 Linux 上，Docker 命令前面需加 `sudo`。若要在没有 `sudo` 情况下运行 Docker 命令，请创建 `docker` 组并添加用户。更多详情，请参阅 [Linux 安装后步骤](https://docs.docker.com/install/linux/linux-postinstall/)。

### 拉取 Docker镜像

```shell
$ sudo docker pull arctern:arctern-spark:latest
```
或者自行构建docker images

```shell
$ cd docker/spark/cpu/
$ ln -s <Arctern 编译安装路径> ./arctern
$ ln -s ../../../python .
$ ln -s ../../../spark/pyspark .
$ sudo docker build -t <image name>:<tag> .
```

## 配置NVIDIA Docker （可选）

### 确认NVIDIA Docker状态
如果你需要运行Arctern全功能版本，需[安装 NVIDIA Docker Version 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))

```shell
$ nvidia-docker version
NVIDIA Docker: 2.0.3
```
### 设置默认runtime

编译/etc/docker/daemon.json，并添加"default-runtime"，配置如下:

```
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
重新加载docker

```
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

## 配置Docker compose
在您的宿主机上[安装Docker compose](https://docs.docker.com/compose/install/)

### 确认Docker compose版本信息

```shell
$ docker-compose version
```

### 修改docker-compose.yml文件

检查docker-compose.yml中image是否填写的是当前您要使用的docker images。
检查master和worker的环境变量设置，[具体设置](https://spark.apache.org/docs/latest/spark-standalone.html)
