# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_definitions(-DUSE_PHYNPU=1)

file(GLOB_RECURSE onnxruntime_providers_phynpu_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/phynpu/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/phynpu/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/phynpu/builders/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/phynpu/builders/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
  "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
)

set(PHYNPU_DDK_PATH ${ONNXRUNTIME_ROOT}/core/providers/phynpu/phytium_npu_sdk)
message("PHYNPU_DDK_PATH is: ${PHYNPU_DDK_PATH}")

if (NOT PHYNPU_DDK_PATH)
  message("PHYNPU_DDK_PATH required for onnxruntime_USE_PHYNPU")
endif()
set(PHYNPU_DDK_INCLUDE_DIR ${PHYNPU_DDK_PATH}/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  message("PHYNPU_DDK_LIB_DIR is: ${PHYNPU_DDK_PATH}/lib/arm64")
  set(PHYNPU_DDK_LIB_DIR ${PHYNPU_DDK_PATH}/lib/arm64)
else()
  message("PHYNPU_DDK_LIB_DIR is: ${PHYNPU_DDK_PATH}/lib/x64")
  set(PHYNPU_DDK_LIB_DIR ${PHYNPU_DDK_PATH}/lib/x64)
endif()

set(PHY_DNN_SO ${PHYNPU_DDK_LIB_DIR}/libphydnn.so)
if(EXISTS ${PHY_DNN_SO})
  message(STATUS "Found libphydnn.so: ${PHY_DNN_SO}")
else()
  message(FATAL_ERROR "libphydnn.so not found in ${PHYNPU_DDK_LIB_DIR}")
endif()

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_phynpu_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_phynpu ${onnxruntime_providers_phynpu_cc_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_phynpu
  onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface
)

file(GLOB LIB_SO_FILES "${PHYNPU_DDK_LIB_DIR}/*.so")
target_link_libraries(onnxruntime_providers_phynpu ${LIB_SO_FILES})
add_dependencies(onnxruntime_providers_phynpu onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_providers_phynpu PROPERTIES FOLDER "ONNXRuntime")
target_include_directories(onnxruntime_providers_phynpu PRIVATE
  ${ONNXRUNTIME_ROOT}  ${PHYNPU_DDK_INCLUDE_DIR}
)
set_target_properties(onnxruntime_providers_phynpu PROPERTIES LINKER_LANGUAGE CXX)

if (NOT onnxruntime_BUILD_SHARED_LIB)
install(TARGETS onnxruntime_providers_phynpu
        ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
        FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
