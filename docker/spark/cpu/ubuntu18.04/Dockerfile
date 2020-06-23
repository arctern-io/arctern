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
    wget -qO- "http://mirror.bit.edu.cn/apache/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz" | tar --strip-components=1 -xz -C /opt/spark && \
    chown -R root:root /opt/spark

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create -n arctern python=3.7 && \
    conda clean --all -y && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate arctern" >> ~/.bashrc

COPY arctern /tmp/arctern

ENV PATH=/opt/conda/bin:$PATH
RUN /tmp/arctern/install_arctern_conda.sh && rm -rf /tmp/arctern && \
    conda clean --all -y

COPY prebuildfs /
COPY rootfs /
RUN /postunpack.sh

ENV NSS_WRAPPER_GROUP="/opt/spark/tmp/nss_group" \
    NSS_WRAPPER_PASSWD="/opt/spark/tmp/nss_passwd" \
    PATH="/opt/conda/envs/arctern/bin:/opt/spark/bin:/opt/spark/sbin:$PATH" \
    SPARK_HOME="/opt/spark"

WORKDIR /opt/spark
USER 1001

# use login shell to activate environment un the RUN commands
SHELL [ "/bin/bash", "-c", "-l" ]

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/run.sh" ]
