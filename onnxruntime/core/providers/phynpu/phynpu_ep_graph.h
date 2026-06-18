 
 

#pragma once

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <phydnn.h>
#include <ife.h>
#include "builders/op_builder.h"


namespace onnxruntime {
namespace phytium_npu {

struct GraphIOInfo {
  std::string name;
  bool is_initializer;
  std::shared_ptr<phydnn_tensor> tensor;
  TensorShape shape;
};

struct GrapOutInfo {
  phydnn_memory out_men;
  size_t length;
  void* out_ptr;
};

class GraphPhynpuEP {
 public:
  explicit GraphPhynpuEP(const GraphViewer& graph_viewer);
  ~GraphPhynpuEP();
  bool Prepare();

  static bool SupportedOp(const GraphViewer& graph_viewer,
                          const NodeUnit& node_unit);

  static bool IsNodeSupportedInGroup(const NodeUnit& node_unit, const GraphViewer& graph_viewer);

  const NodeUnit& GetNodeUnit(const Node* node) const;

  bool& GetCompiled() { return compiled_; }
  phydnn_network GetGraph() { return graph_; }  // GetNetwork()
  const GraphViewer* GetGraphViewer() { return &graph_viewer_; }

  std::map<std::string, std::shared_ptr<phydnn_tensor>>& GetTensors() {
    return tensors_;
  }

  std::vector<std::shared_ptr<GraphIOInfo>>& GetGraphInputs() {
    return graph_inputs_;
  }

  std::vector<std::shared_ptr<GraphIOInfo>>& GetGraphOutputs() {
    return graph_outputs_;
  }

  // Tensor
  phydnn_tensor CreateInputTensor(phydnn_tensor_descriptor *desc, bool is_qtensor);
  phydnn_tensor CreateFixedInputTensor(phydnn_tensor_descriptor *desc,
                                       const void *const data,
                                       bool copy);

  // Network
  bool CheckConfigFileExists(const std::string &map_config_path);
  phydnn_network_object CreateNetworkObject(unsigned int num_inputs,
                                            phydnn_tensor *inputs,
                                            unsigned int num_outputs,
                                            phydnn_tensor *outputs,
                                            bool is_quant = false);
  void ExecuteNetworkObject(bool blocking_execute,
                            unsigned int num_events_in_wait_list,
                            const phydnn_event event_wait_list[],
                            phydnn_event *event);
  void GetNetworkObjectInputs(unsigned int max_inputs,
                              phydnn_input inputs[],
                              unsigned int *num_inputs);
  void GetNetworkObjectOutputs(unsigned int max_outputs,
                               phydnn_output outputs[],
                               unsigned int *num_outputs);
  phydnn_tensor_descriptor GetInputDescriptor(phydnn_input input);
  phydnn_tensor_descriptor GetOutputDescriptor(phydnn_output output);
  phydnn_tensor_descriptor GetTensorDescriptor(phydnn_tensor tensor);
  size_t GetDescriptorSize(const phydnn_tensor_descriptor *const descriptor);
  void AddBindingInput(phydnn_input input, phydnn_memory memory);
  void AddBindingOutput(phydnn_output output, phydnn_memory memory);

  // Memory
  phydnn_memory ImportMemory(
      void *buffer,
      size_t size,
      phydnn_import_mem_type import_mem_type = PHYDNN_IMPORT_MEM_TYPE_CPU);
  phydnn_memory AllocateMemory(size_t size);
  void DestroyMemory(phydnn_memory memory);
  void *LockMemory(phydnn_memory memory, phydnn_lock_access lock_access);
  void UnlockMemory(phydnn_memory memory);
  std::shared_ptr<phydnn_tensor> UpdateIntputTensor(
    const onnxruntime::GraphViewer& graph_viewer, const NodeUnitIODef nudef, const NodeUnit& node_unit);
  std::shared_ptr<phydnn_tensor> UpdateOutputTensor(
    const NodeUnitIODef nudef, phydnn_tensor tensor);
  uint8_t *GetBuffer(size_t size);
  phydnn_tensor CreatePhynpuTensor(phydnn_tensor_descriptor desc,
                            const ONNX_NAMESPACE::TensorProto* tensor_proto,
                            ONNX_NAMESPACE::TensorProto_DataType data_type,
                            size_t size,
                            bool is_qtensor);
  template <typename T>
  phydnn_tensor QuantizeLinear(phydnn_tensor_descriptor desc,
                               const std::vector<T>& input);
  template <typename T>
  phydnn_tensor QuantizeLinear(phydnn_tensor_descriptor desc,
                                           float scale,
                                           int32_t zero_point,
                                           const std::vector<T>& input);
 public:
 // 图中间变量
  std::vector<std::shared_ptr<GraphIOInfo>> graph_inputs_;
  std::vector<std::shared_ptr<GraphIOInfo>> graph_outputs_;
  std::vector<phydnn_input> input_info_;
  std::vector<phydnn_output> output_info_;
  std::vector<std::pair<phydnn_memory, size_t>> input_memory_;
  std::vector<GrapOutInfo> output_memory_;  // 此次可以修改为shared_ptr方式
  std::map<int, OrtValue> node_output_map;

 private:
 // 图或节点操作自定义的成员接口
  std::map<std::string, std::shared_ptr<phydnn_tensor>> tensors_;
  phydnn_device device_;
  phydnn_network graph_{nullptr};   // network_
  phydnn_context context_{nullptr};
  phydnn_binding binding_{nullptr};
  phydnn_network_object network_object_{nullptr};

// 存储图结构
  const GraphViewer& graph_viewer_;

// 存储图中NodeUnit信息，图中的节点信息，包含了算子的操作属性
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder_;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map_;
  std::vector<std::unique_ptr<uint8_t[]>> buffers_;

// 指明节点是否已编译，如果为false，则说明该节点未被编译
  bool compiled_;
};

}  // namespace phytium_npu
}  // namespace onnxruntime
