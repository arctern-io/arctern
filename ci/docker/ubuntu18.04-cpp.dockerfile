FROM ubuntu:18.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget ca-certificates \
    git ccache lcov clang-6.0 clang-format-6.0 clang-tidy-6.0 \
    libgl-dev libosmesa6-dev libglu1-mesa-dev  && \
    wget -qO- "https://cmake.org/files/v3.14/cmake-3.14.3-Linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

COPY ci/yaml/conda_env_cpp.yml /arctern/ci/yaml/

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda update --all -y && \
    conda create -n arctern -c conda-forge -q \
    --file /arctern/ci/yaml/conda_env_cpp.yml && \
    conda clean --all -y && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV RUN_LINT=ON \
    BUILD_UNITTEST=ON \
    BUILD_COVERAGE=ON \
    USE_GPU=OFF

# use login shell to activate arrow environment un the RUN commands
SHELL [ "/bin/bash", "-c", "-l" ]

# use login shell when running the container
ENTRYPOINT [ "/bin/bash", "-c", "-l" ]
