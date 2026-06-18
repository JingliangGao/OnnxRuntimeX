// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//Data: 2026-01-08 

#pragma once

#include <memory>

#include "core/providers/providers.h"

namespace onnxruntime {
struct PHYnpuProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create();
};
}  // namespace onnxruntime
