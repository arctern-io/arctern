FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends gosu sudo \
    wget libnss-wrapper openjdk-8-jdk && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

COPY zgis.yml /tmp/zgis.yml

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda env create -f /tmp/zgis.yml && \
    rm /tmp/zgis.yml && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate zgis_dev" >> ~/.bashrc

RUN mkdir -p /opt/spark && \
    wget -qO- "http://mirror.bit.edu.cn/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop2.7.tgz" | tar --strip-components=1 -xz -C /opt/spark && \
    chown -R root:root /opt/spark

COPY arctern /var/lib/arctern
COPY python /tmp/python

RUN source /var/lib/arctern/scripts/arctern_env.sh && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate zgis_dev && \
    cd /tmp/python && \
    python setup.py build build_ext --library-dirs="$ARCTERN_LIB_DIR" && \
    python setup.py install && \
    rm -rf /tmp/python

COPY pyspark /tmp/pyspark

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate zgis_dev && \
    cd /tmp/pyspark && \
    python setup.py build && \
    python setup.py install && \
    rm -rf /tmp/pyspark

COPY prebuildfs /
COPY rootfs /
RUN /postunpack.sh

ENV NSS_WRAPPER_GROUP="/opt/spark/tmp/nss_group" \
    NSS_WRAPPER_PASSWD="/opt/spark/tmp/nss_passwd" \
    PATH="/opt/conda/envs/zgis_dev/bin:/opt/spark/bin:/opt/spark/sbin:$PATH" \
    SPARK_HOME="/opt/spark"

WORKDIR /opt/spark
USER 1001
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/run.sh" ]
