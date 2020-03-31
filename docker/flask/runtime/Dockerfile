ARG IMAGE_NAME
FROM ${IMAGE_NAME}:base

COPY gui/server /arctern-gui

WORKDIR /arctern-gui

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate arctern && \
    cd /arctern-gui && \
    pip install -r requirements.txt

# use login shell to activate arrow environment un the RUN commands
SHELL [ "/bin/bash", "-c", "-l" ]

# use login shell when running the container
ENTRYPOINT [ "/bin/bash", "-c", "-l", "python", "manage.py", "-r"]
