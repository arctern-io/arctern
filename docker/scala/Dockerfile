FROM ubuntu:18.04

RUN apt update && apt install -y openjdk-8-jre curl gnupg2
RUN echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
RUN curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add
RUN apt update && apt install -y sbt && \
    apt-get remove --purge -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /scala

CMD ["/usr/bin/sbt", "test"]
