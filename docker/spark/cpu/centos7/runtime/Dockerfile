ARG IMAGE_NAME
FROM ${IMAGE_NAME}:centos7-base

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
