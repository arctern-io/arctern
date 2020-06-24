FROM nvidia/cudagl:10.0-devel-centos7

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN yum install -y epel-release centos-release-scl-rh && yum install -y wget curl && \
    wget -qO- "https://cmake.org/files/v3.14/cmake-3.14.3-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    yum install -y glibc-headers gcc-c++ ccache lcov make automake bzip2 git libcurl-devel \
    llvm-toolset-7.0-clang llvm-toolset-7.0-clang-tools-extra && \
    mkdir /build && \
    wget -qO- "https://mirrors.ustc.edu.cn/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz" | tar --strip-components=1 -xz -C /build && \
    cd build && ./contrib/download_prerequisites && \
    ./configure --enable-checking=release --enable-languages=c,c++ --disable-multilib && \
    make && make install && \
    rm -rf /build && \
    rm -rf /var/cache/yum/*

RUN echo "source scl_source enable llvm-toolset-7.0" >> /etc/profile.d/llvm-toolset-7.sh

ENV CLANG_TOOLS_PATH /opt/rh/llvm-toolset-7.0/root/usr/bin

COPY docker/build_env/build_conda_env.yml /tmp/build_conda_env.yml

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda update --all -y && \
    conda env create -f /tmp/build_conda_env.yml && \
    conda clean --all -y && \
    rm /tmp/build_conda_env.yml && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

COPY docker/build_env/docker-entrypoint.sh /opt/docker-entrypoint.sh
WORKDIR /root

# use login shell to activate environment un the RUN commands
SHELL [ "/bin/bash", "-c", "-l" ]

# use login shell when running the container
ENTRYPOINT [ "/bin/bash", "-c", "-l" ]
