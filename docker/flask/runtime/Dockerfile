ARG IMAGE_NAME
FROM ${IMAGE_NAME}:base

COPY python /arctern/python

RUN cd /arctern/python && \
    pip3 --no-cache-dir install --upgrade cython pyarrow && \
    python setup.py build build_ext --issymbol && \
    python setup.py install

COPY spark /arctern/spark

RUN cd /arctern/spark/pyspark && \
    python setup.py build && \
    python setup.py install

COPY gui/server /arctern/gui/server

RUN cd /arctern/gui/server && \
    pip3 --no-cache-dir install -r requirements.txt

EXPOSE 8080

WORKDIR /arctern/gui/server

# use login shell when running the container
CMD ["python", "manage.py", "-r"]
