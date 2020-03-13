ARG base
FROM ${base}

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda update --all -y && \
    conda install -n arctern -c conda-forge -q \
    --file /arctern/ci/yaml/conda_env_python.yml && \
    conda clean --all -y
