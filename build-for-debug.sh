#!/bin/bash

CURRENT_DIR=$(pwd)
UNIT_TESTS=ON
BUILD_FOLDER="build"
ONNX_ROOT="/home/kylin/gjl/project/onnx_depend/onnx"

clear

SUDO_ER=''
if [ "$EUID" -ne 0 ]; then
    SUDO_ER='sudo'
else
    SUDO_ER=''
fi

if [ -e "build.ninja" ]; then
    rm -rf _deps
    rm -rf build
    rm build.ninja
    rm build.ninja*
    rm *.log
    rm *.onnx
    rm CMakeFiles.txt
    rm .ninja_deps 
    rm .ninja_log
    rm CMakeCache.txt
    rm -rf CMakeFiles/
    rm CTestTestfile.cmake 
    rm PROJECT_CONFIG_FILE
    rm build*.log
    rm debian/debhelper-build-stamp
    rm debian/files
    rm debian/libonnxruntime-dev.substvars
    rm debian/libonnxruntime.substvars
    rm debian/libonnxruntime-dev.debhelper.log
    rm debian/libonnxruntime.debhelper.log
    rm -rf debian/.debhelper/
    rm -rf debian/debhelper-build-stamp/
    rm -rf debian/tmp/
    rm -rf debian/libonnxruntime-dev/
    rm -rf debian/libonnxruntime/
    rm -rf reproducible-path/
    rm generated_source.c
    rm generated_source.cinstall_manifest.txt
    rm DartConfiguration.tcl
    rm cmake_install.cmake 
    rm install_manifest.txt 
    rm libonnxruntime.pc
    rm libonnxruntime.so
    rm onnxruntimeConfig.cmake
    rm onnxruntimeConfigVersion.cmake
    rm onnxruntime_mlas_q4dq
    rm onnxruntime_config.h
    rm libonnxruntime.so.1.20.1
    rm onnxruntime.lds
    rm libonnxruntime_providers_shared.so
    rm libonnxruntime_*.a
    rm compile_commands.json
fi

cd ${CURRENT_DIR}
if [ "$1" == "local" ]; then
    echo "[INFO] start build locally ... "
    if [ -d ${BUILD_FOLDER} ]; then
        rm -rf ${BUILD_FOLDER}
    fi
    mkdir ${BUILD_FOLDER}
    cd ${BUILD_FOLDER}

    cmake ../cmake \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_INCLUDEDIR=include \
        -Deigen_SOURCE_PATH=/usr/include/eigen3 \
        -Donnxruntime_BUILD_BENCHMARKS=OFF \
        -Donnxruntime_BUILD_SHARED_LIB=ON \
        -Donnxruntime_BUILD_UNIT_TESTS=${UNIT_TESTS} \
        -Donnxruntime_DISABLE_ABSEIL=ON \
        -Donnxruntime_ENABLE_CPUINFO=ON \
        -Donnxruntime_ENABLE_PYTHON=OFF \
        -Donnxruntime_ONNX_TEST_DATA_DIR=/usr/share/libonnx-testdata/data \
        -Donnxruntime_USE_FULL_PROTOBUF=ON \
        -Donnxruntime_USE_PREINSTALLED_EIGEN=ON \
        -Donnxruntime_DISABLE_CONTRIB_OPS=OFF \
        -Donnxruntime_USE_PHYNPU=OFF \
        -Donnx_SOURCE_DIR=${ONNX_ROOT} \
        -Donnxruntime_ONNX_PROTO_DATA_DIR=${ONNX_ROOT} \
        -DCMAKE_CXX_FLAGS="-Wno-error=unused-result -Wno-error=range-loop-construct -I${ONNX_ROOT}/onnx" \
        -DCMAKE_INCLUDE_PATH=${ONNX_ROOT}/onnx:${CMAKE_INCLUDE_PATH} \
        -DPROTOBUF_IMPORT_DIRS=${ONNX_ROOT}/onnx

    echo -e "
        ****************** CMake Configuration ******************
        cmake ../cmake \\
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \\
            -DCMAKE_INSTALL_INCLUDEDIR=include \\
            -Deigen_SOURCE_PATH=/usr/include/eigen3 \\
            -Donnxruntime_BUILD_BENCHMARKS=OFF \\
            -Donnxruntime_BUILD_SHARED_LIB=ON \\
            -Donnxruntime_BUILD_UNIT_TESTS=${UNIT_TESTS} \\
            -Donnxruntime_DISABLE_ABSEIL=ON \\
            -Donnxruntime_ENABLE_CPUINFO=ON \\
            -Donnxruntime_ENABLE_PYTHON=OFF \\
            -Donnxruntime_ONNX_TEST_DATA_DIR=/usr/share/libonnx-testdata/data \\
            -Donnxruntime_ONNX_PROTO_DATA_DIR=${ONNX_ROOT} \\
            -Donnxruntime_USE_FULL_PROTOBUF=ON \\
            -Donnxruntime_USE_PREINSTALLED_EIGEN=ON \\
            -Donnxruntime_DISABLE_CONTRIB_OPS=OFF \\
            -Donnxruntime_USE_PHYNPU=OFF \\
            -Donnx_SOURCE_DIR=${ONNX_ROOT} \\
            -DCMAKE_CXX_FLAGS=\"-Wno-error=unused-result -Wno-error=range-loop-construct -I${ONNX_ROOT}/onnx\" \\
            -DCMAKE_INCLUDE_PATH=${ONNX_ROOT}/onnx:${CMAKE_INCLUDE_PATH} \\
            -DPROTOBUF_IMPORT_DIRS=${ONNX_ROOT}/onnx
        **********************************************************
    "

    echo "[INFO] start build ..."
    make -j$(nproc)
else
    echo "[INFO] start build deb package ... "
    dpkg-buildpackage -T clean
    dpkg-buildpackage -us -uc -b 
fi

echo "[INFO] All done!"