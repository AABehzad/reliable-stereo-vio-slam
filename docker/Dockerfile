FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------
# Base development tools
# --------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    git \
    gdb \
    pkg-config \
    curl \
    wget \
    ca-certificates \
    clang-format \
    \
    libopencv-dev \
    libeigen3-dev \
    libyaml-cpp-dev \
    \
    libceres-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsuitesparse-dev \
    \
    libgtest-dev \
    \
    python3 \
    python3-pip \
    python3-venv \
    python3-numpy \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

ENV CMAKE_GENERATOR=Ninja

WORKDIR /workspace

CMD ["/bin/bash"]