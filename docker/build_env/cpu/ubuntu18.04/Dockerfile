FROM ubuntu:18.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget ca-certificates curl \
    git ccache lcov clang-6.0 clang-format-6.0 clang-tidy-6.0 \
    libgl-dev libosmesa6-dev libglu1-mesa-dev  && \
    wget -qO- "https://cmake.org/files/v3.14/cmake-3.14.3-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

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

# use login shell to activate environment un the RUN commands
SHELL [ "/bin/bash", "-c", "-l" ]

# use login shell when running the container
ENTRYPOINT [ "/bin/bash", "-c", "-l" ]
