FROM ubuntu:18.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates openjdk-8-jdk \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda update --all -y && \
    conda create -n arctern python=3.7 && \
    conda clean --all -y && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate arctern" >> ~/.bashrc

RUN mkdir -p /opt/spark && \
    wget -qO- "http://mirror.bit.edu.cn/apache/spark/spark-3.0.0-preview2/spark-3.0.0-preview2-bin-hadoop2.7.tgz" | tar --strip-components=1 -xz -C /opt/spark && \
    chown -R root:root /opt/spark && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate arctern && \
    cd /opt/spark/python && \
    python setup.py install && \
    cd / && rm -rf /opt/spark && \
    conda install -y -n arctern -c conda-forge -c arctern-dev arctern-spark && \
    conda clean --all -y
