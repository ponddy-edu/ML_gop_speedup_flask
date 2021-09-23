FROM ubuntu:18.04
ENV LANG C.UTF-8
RUN apt-get update && apt-get -y install python3 gnupg2 wget

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | apt-key add - && \
    echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list && \
    apt-get update && \
    apt-get -y install intel-mkl-64bit-2019.5-075

WORKDIR /home/ubuntu/gopserver
COPY mixedZhuyin mixedZhuyin
COPY kaldi_md kaldi_rt
COPY server.py ./

EXPOSE 8085
CMD python3 server.py
