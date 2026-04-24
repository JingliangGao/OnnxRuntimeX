#!/bin/bash

clear

# check if force remove build directory
if [ "$1" == "force" ]; then
  if [ -d "build" ]; then
    echo ">>[INFO]: Removing existing build directory..."
    rm -rf build
  fi
fi


# check if sudoers
if [ "$EUID" -ne 0 ]; then
    SUDO_ER='sudo'
else
    SUDO_ER=''
fi

echo ">>[INFO]: Remove useless packages ..."
${SUDO_ER} apt remove libonnx-dev -y

# build
echo ">>[INFO]: Building ... "
./build.sh --config RelWithDebInfo \
           --build_shared_lib \
           --parallel \
           --compile_no_warning_as_error \
           --allow_running_as_root \
           --skip_tests \
           --compile_no_warning_as_error \
           --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
           --cmake_extra_defines ONNX_USE_PROTOBUF_SHARED_LIBS=OFF \
           --cmake_extra_defines onnxruntime_USE_FULL_PROTOBUF=ON \
           --cmake_extra_defines onnxruntime_ENABLE_CPU_FP16_OPS=OFF \
           --cmake_extra_defines onnxruntime_ENABLE_AVX512=OFF \
           --cmake_extra_defines CMAKE_CXX_STANDARD=20
