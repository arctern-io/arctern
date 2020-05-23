FROM python:3.6.8-jessie

COPY gui/server /arctern/gui/server

RUN cd /arctern/gui/server/arctern_server && \
    python3 -m pip --no-cache-dir install -r requirements.txt

EXPOSE 8080

WORKDIR /arctern/gui/server/arctern_server

# use login shell when running the container
ENV PYTHONPATH=/arctern/gui/server:$PYTHONPATH
CMD ["python", "manage.py", "-r"]
