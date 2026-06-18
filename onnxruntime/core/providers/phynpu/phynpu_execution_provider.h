 
//Data: 2026-01-08 

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

class PHYnpuExecutionProvider : public IExecutionProvider {
public:
  PHYnpuExecutionProvider();
  virtual ~PHYnpuExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const IKernelLookup& /*kernel_lookup*/) const override;
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
  OrtMutex& GetMutex() { return mutex_; }

public:
  std::map<int, OrtValue> input_id_map;
  std::set<int> input_args_vec;
  
private:
  OrtMutex mutex_;
  
};

}  // namespace onnxruntime
