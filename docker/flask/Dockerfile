FROM ubuntu:18.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends build-essential wget ca-certificates openjdk-8-jdk \
    python3-pip python3.7-dev python3-setuptools \
    libgl-dev libosmesa6-dev libglu1-mesa-dev && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3.7 python && \
    python3.7 -m pip --no-cache-dir install --upgrade pip && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/spark && \
    wget -qO- "http://mirror.bit.edu.cn/apache/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz" | tar --strip-components=1 -xz -C /opt/spark && \
    chown -R root:root /opt/spark && \
    cd /opt/spark/python && \
    python setup.py install && \
    cd / && rm -rf /opt/spark

COPY python /arctern/python

RUN cd /arctern/python && \
    python3.7 -m pip --no-cache-dir install --upgrade cython pyarrow && \
    python setup.py build build_ext --issymbol && \
    python setup.py install && \
    cd / && rm -rf /arctern/python

COPY spark /arctern/spark

RUN cd /arctern/spark/pyspark && \
    python setup.py build && \
    python setup.py install && \
    cd / && rm -rf /arctern/spark

COPY gui/server/arctern_server /arctern/gui/server/arctern_server

RUN cd /arctern/gui/server/arctern_server && \
    python3.7 -m pip --no-cache-dir install -r requirements.txt

EXPOSE 8080

WORKDIR /arctern/gui/server/arctern_server

# use login shell when running the container
ENV PYTHONPATH=/arctern/gui/server:$PYTHONPATH
CMD ["python", "manage.py", "-r"]
