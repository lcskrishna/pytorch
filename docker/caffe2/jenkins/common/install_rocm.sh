#!/bin/bash

set -ex

install_ubuntu() {
    apt-get update
    apt-get install -y wget
    apt-get install -y libopenblas-dev

    DEB_ROCM_REPO=http://repo.radeon.com/rocm/misc/facebook/apt/.apt_1.9.white_rabbit/debian
    # Add rocm repository
    wget -qO - $DEB_ROCM_REPO/rocm.gpg.key | apt-key add -
    echo "deb [arch=amd64] $DEB_ROCM_REPO xenial main" > /etc/apt/sources.list.d/rocm.list
    apt-get update --allow-insecure-repositories

    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
                   rocm-dev \
                   rocm-libs \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   miopengemm \
                   rocblas \
                   rocm-profiler \
                   cxlactivitylogger \
                   rocsparse \
                   hipsparse \
                   rocrand \
                   hip-thrust

    # HIP has a bug that drops DEBUG symbols in generated MakeFiles.
    # https://github.com/ROCm-Developer-Tools/HIP/pull/588
    if [[ -f /opt/rocm/hip/cmake/FindHIP.cmake ]]; then
        sudo sed -i 's/set(_hip_build_configuration "${CMAKE_BUILD_TYPE}")/string(TOUPPER _hip_build_configuration "${CMAKE_BUILD_TYPE}")/' /opt/rocm/hip/cmake/FindHIP.cmake
    fi
}

install_centos() {
    echo "Not implemented yet"
    exit 1
}
 
# Install Python packages depending on the base OS
if [ -f /etc/lsb-release ]; then
  install_ubuntu
elif [ -f /etc/os-release ]; then
  install_centos
else
  echo "Unable to determine OS..."
  exit 1
fi
