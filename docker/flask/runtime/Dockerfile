ARG IMAGE_NAME
FROM ${IMAGE_NAME}:base

COPY python /pyarctern

RUN cd /pyarctern && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate arctern && \
    python setup.py build build_ext --issymbol && \
    python setup.py install

COPY spark /arctern-spark

RUN cd /arctern-spark/pyspark && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate arctern && \
    python setup.py build && \
    python setup.py install

COPY gui/server /arctern-gui

WORKDIR /arctern-gui

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate arctern && \
    cd /arctern-gui && \
    pip install -r requirements.txt

# use login shell to activate arrow environment un the RUN commands
SHELL [ "/bin/bash", "-c", "-l" ]

EXPOSE 8080

ENV PATH /opt/conda/bin:$PATH

# Add Tini
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

ENV PATH /opt/conda/bin:$PATH

ENTRYPOINT ["/tini", "--"]

# use login shell when running the container
CMD ["conda", "run", "-n", "python", "manage.py", "-r"]
