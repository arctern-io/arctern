FROM ubuntu:16.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget ca-certificates \
    git ccache lcov clang-format-6.0 clang-tidy-6.0 libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev libglu1-mesa-dev  && \
    wget -qO- "https://cmake.org/files/v3.14/cmake-3.14.3-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

COPY zgis.yml /tmp/zgis.yml

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
#    wget --quiet https://raw.githubusercontent.com/zilliztech/GIS/conda/zgis.yml -O /tmp/zgis.yml && \
    conda env create -f /tmp/zgis.yml && \
    rm /tmp/zgis.yml && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate zgis_dev" >> ~/.bashrc

COPY docker-entrypoint.sh /opt/docker-entrypoint.sh
WORKDIR /root

ENTRYPOINT [ "/opt/docker-entrypoint.sh" ]
CMD [ "start" ]
