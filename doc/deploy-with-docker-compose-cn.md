# 使用 docker compose 部署 Arctern

## 安装前提

### 系统要求


| 操作系统    | 版本          |
| ---------- | ------------ |
| CentOS     | 7 或以上      |
| Ubuntu LTS | 16.04 或以上  |

### 软件要求


| 软件名称        | 版本          | 备注  |
| ----------     | ------------ | ----- |
| Docker         | 17.06.0 或以上| 必要  |
| Docker compose | 1.17.1 或以上 | 必要  |
| Nvidia Docker  | Version 2    | 可选  |

## 配置Docker

### 确认Docker 运行状态

通过以下命令确认 Docker daemon 运行状态：

```shell
$ docker info
```

如果上述命令未能正常打印 Docker 相关信息，请启动 **Docker** daemon.

> 提示：在 Linux 环境下，Docker 命令需要 `sudo` 权限。如需要不使用 `sudo` 权限下运行 Docker 命令，请创建 `docker` 组并添加用户。详情请参阅 [Linux 安装后步骤](https://docs.docker.com/install/linux/linux-postinstall/)。

### 获取 Docker 镜像

使用以下命令拉取 Docker 镜像：

```shell
$ sudo docker pull arctern:arctern-spark:latest
```

或者使用下述命令构建Docker镜像：

CPU 版本
```shell
$ pushd docker/spark/cpu/base
$ sudo docker build -t arctern-spark:ubuntu18.04-base .
$ popd
$ pushd docker/spark/cpu/runtime
$ sudo docker build -t arctern-spark:ubuntu18.04-runtime --build-arg 'IMAGE_NAME=arctern-spark' .
```

GPU 版本
```shell
$ pushd docker/spark/gpu/base
$ sudo docker build -t arctern-spark-gpu:ubuntu18.04-base .
$ popd
$ pushd docker/spark/gpu/runtime
$ sudo docker build -t arctern-spark-gpu:ubuntu18.04-runtime --build-arg 'IMAGE_NAME=arctern-spark-gpu' .
```

## 配置 NVIDIA Docker （可选）

### 确认 NVIDIA Docker状态
如果需要运行 GPU 版本 Arctern，需[安装 NVIDIA Docker Version 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))。

通过以下命令确认 NVIDIA Docker 是否安装成功。

```shell
$ nvidia-docker version
NVIDIA Docker: 2.0.3
```

### 设置默认运行时环境

编辑`/etc/docker/daemon.json`文件，并添加"default-runtime"相关配置:

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
使用以下命令重新加载docker：

```shell
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

## 配置 Docker compose

[安装Docker compose](https://docs.docker.com/compose/install/)，并通过以下命令确认 Docker compose 版本信息

```shell
$ docker-compose version
```

### 修改 docker-compose.yml 文件

检查docker-compose.yml中image字段是否正确(arctern-spark:ubuntu18.04-runtime 或 arctern-spark-gpu:ubuntu18.04-runtime)。
并[检查 master 和 worker 的环境变量设置是否正确](https://spark.apache.org/docs/latest/spark-standalone.html)。

### 启动分布式集群

前台执行
```shell
$ sudo docker-compose up
```

后台执行
```shell
$ sudo docker-compose up -d
```

### 关闭分布式集群

```shell
$ sudo docker-compose down
```
