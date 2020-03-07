FROM ubuntu:18.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends gosu sudo \
    build-essential wget ca-certificates libnss-wrapper openjdk-8-jdk \
    libgl-dev libosmesa6-dev libglu1-mesa-dev  && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/spark && \
    wget -qO- "http://mirror.bit.edu.cn/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop2.7.tgz" | tar --strip-components=1 -xz -C /opt/spark && \
    chown -R root:root /opt/spark

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n arctern python=3.7 && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate arctern" >> ~/.bashrc

RUN mkdir -p /home/arctern_test

COPY arctern /tmp/arctern

ENV PATH=/opt/conda/bin:$PATH
RUN /tmp/arctern/install_arctern_spark.sh && rm -rf /tmp/arctern

COPY prebuildfs /
COPY rootfs /
RUN /postunpack.sh

ENV NSS_WRAPPER_GROUP="/opt/spark/tmp/nss_group" \
    NSS_WRAPPER_PASSWD="/opt/spark/tmp/nss_passwd" \
    PATH="/opt/conda/envs/zgis_dev/bin:/opt/spark/bin:/opt/spark/sbin:$PATH" \
    SPARK_HOME="/opt/spark"

COPY docker-entrypoint.sh /entrypoint.sh

WORKDIR /opt/spark
USER 1000
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/run.sh" ]
