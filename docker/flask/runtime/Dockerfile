ARG IMAGE_NAME
FROM ${IMAGE_NAME}:base

COPY python /pyarctern

RUN cd /pyarctern && \
    pip3 --no-cache-dir install --upgrade cython && \
    python setup.py build build_ext --issymbol && \
    python setup.py install

COPY spark /arctern-spark

RUN cd /arctern-spark/pyspark && \
    python setup.py build && \
    python setup.py install

COPY gui/server /arctern-gui

RUN cd /arctern-gui && \
    pip3 --no-cache-dir install -r requirements.txt

EXPOSE 8080

WORKDIR /arctern-gui

# use login shell when running the container
CMD ["python", "manage.py", "-r"]
