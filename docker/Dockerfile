FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    wget \
    nano \ 
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    freeglut3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    gdb \
    lldb \
    clang \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/tauzn-clock/pylbfgs
WORKDIR /pylbfgs
RUN git submodule update --init --recursive
RUN chmod +x ./install_liblbfgs.sh
RUN ./install_liblbfgs.sh
RUN pip3 install -r requirements.txt --break-system-packages
RUN python3 setup.py build_ext --inplace