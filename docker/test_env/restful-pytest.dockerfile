FROM python:3.7-alpine

RUN pip3 install --no-cache-dir \
    -r gui/server/tests/restful/requirements.txt