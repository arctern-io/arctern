# Using docker compose to deploy Arctern
<!-- TOC -->

- [Prerequisites](#prerequisites)
    - [Operating system requirements](#operating-system-requirements)
    - [Software dependencies](#software-dependencies)
- [Configuring Docker](#configure-docker)
    - [Confirm Docker status](#confirm-docker-status)
    - [Get docker images](#get-docker-images)
- [Configuring NVIDIA Docker (Optional)](#configuring-nvidia-docker-optional)
    - [Confirm NVIDIA Docker status](#confirm-nvidia-docker-status)
    - [Configure default runtime environment](#configure-default-runtime-environment)
- [Configuring Docker compose](#configuring-docker-compose)
    - [Edit docker-compose.yml](#edit-docker-composeyml)
    - [Launch distributed cluster](#launch-distributed-cluster)
    - [Shutdown distributed cluster](#shutdown-distributed-cluster)

<!-- /TOC -->


## Prerequisites

### Operating system requirements


| Operating system   | Version          |
| ---------- | ------------ |
| CentOS     | 7 or higher      |
| Ubuntu LTS | 16.04 or higher  |

### Software dependencies

| Component        | Version          | Required?  |
| ----------     | ------------ | ----- |
| Docker         | 17.06.0 or higher| Yes  |
| Docker compose | 1.17.1 or higher | Yes  |
| Nvidia Docker  | Version 2    | No  |

## Configuring Docker

### Confirm Docker status

Use the following command to confirm the status of docker daemon:

```shell
$ docker info
```

If the command above cannot print docker information, please run **Docker** daemon.

> Note: On Linux, Docker needs sudo privileges. To run Docker command without `sudo`, create the `docker` group and add your user. For details, see the [post-installation steps for Linux](https://docs.docker.com/install/linux/linux-postinstall/).

### Get docker images

Use the following command to pull docker image:

```shell
$ sudo docker pull arctern:arctern-spark:latest
```

Or use the following command to build images:

CPU version

```shell
$ pushd docker/spark/cpu/base
$ sudo docker build -t arctern-spark:ubuntu18.04-base .
$ popd
$ pushd docker/spark/cpu/runtime
$ sudo docker build -t arctern-spark:ubuntu18.04-runtime --build-arg 'IMAGE_NAME=arctern-spark' .
```

GPU version

```shell
$ pushd docker/spark/gpu/base
$ sudo docker build -t arctern-spark-gpu:ubuntu18.04-base .
$ popd
$ pushd docker/spark/gpu/runtime
$ sudo docker build -t arctern-spark-gpu:ubuntu18.04-runtime --build-arg 'IMAGE_NAME=arctern-spark-gpu' .
```

## Configuring NVIDIA Docker (Optional)

### Confirm NVIDIA Docker status

To run Arctern with GPU support, you need to [install NVIDIA Docker Version 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

Use the following command to confirm whether NVIDIA docker is installed:

```shell
$ nvidia-docker version
NVIDIA Docker: 2.0.3
```

### Configure default runtime environment

Edit `/etc/docker/daemon.json` and add  "default-runtime" configuration:

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
Use the following command to reload docker:

```shell
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

## Configuring Docker compose

[Install Docker compose](https://docs.docker.com/compose/install/) and use the following command to confirm Docker compose version info:

```shell
$ docker-compose version
```

### Edit docker-compose.yml

Check whether the image field in docker-compose.yml is correct (arctern-spark:ubuntu18.04-runtime or arctern-spark-gpu:ubuntu18.04-runtime).
Also [check whether environment variables for master and worker are correct](https://spark.apache.org/docs/latest/spark-standalone.html).

### Launch distributed cluster

Frontend
```shell
$ sudo docker-compose up
```

backend
```shell
$ sudo docker-compose up -d
```

### Shutdown distributed cluster

```shell
$ sudo docker-compose down
```
