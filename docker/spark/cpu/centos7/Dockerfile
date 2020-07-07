FROM centos:centos7

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN yum install -y wget centos-release-scl-rh java-1.8.0-openjdk \
    mesa-libGLU-devel mesa-libOSMesa-devel  && \
    yum --enablerepo=centos-sclo-rh-testing install -y nss_wrapper && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    rm -rf /var/cache/yum/*

# grab gosu for easy step-down from root
ENV GOSU_VERSION 1.11
RUN set -x \
    && yum install -y wget && yum clean all \
    && wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(if [ `arch` = 'x86_64' ]; then echo 'amd64'; else echo `arch`; fi)" \
    && wget -O /usr/local/bin/gosu.asc "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(if [ `arch` = 'x86_64' ]; then echo 'amd64'; else echo `arch`; fi).asc" \
    && export GNUPGHOME="$(mktemp -d)" \
    && gpg --keyserver ha.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4 \
    && gpg --batch --verify /usr/local/bin/gosu.asc /usr/local/bin/gosu \
    && rm -r "$GNUPGHOME" /usr/local/bin/gosu.asc \
    && chmod +x /usr/local/bin/gosu \
    && gosu nobody true \
    && rm -rf /var/cache/yum/*

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
