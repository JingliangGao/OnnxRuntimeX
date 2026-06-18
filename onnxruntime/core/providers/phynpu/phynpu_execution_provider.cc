// Copyright 2020 phytium-chips.com Inc.
//Data: 2026-01-08 

#include <unistd.h>
#include <limits>
#include <set>
#include <unordered_set>
#include <map>
#include <list>
#include <future>
#include <functional>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "core/session/inference_session.h"
#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/node_unit.h"
#include "core/providers/phynpu/builders/op_builder.h"
#include "core/providers/phynpu/builders/op_builder_factory.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/partitioning_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "phynpu_execution_provider.h"
#include "phynpu_ep_graph.h"

using std::string;
using std::vector;

namespace onnxruntime {

constexpr const char* PHYNPU = "PHYNPU";
const std::string PHYNPU_UNSPPORTED_NODES_FILE = "phynpu_unspported_nodes_file.txt";

PHYnpuExecutionProvider::PHYnpuExecutionProvider()
  : IExecutionProvider{onnxruntime::kPHYNPUExecutionProvider} {
    input_id_map.clear();
    input_args_vec.clear();
}

PHYnpuExecutionProvider::~PHYnpuExecutionProvider() {
  input_id_map.clear();
  input_args_vec.clear();
}

bool WriteFile(std::string fname, std::map<size_t, std::string> *m) {
  int count = 0;
  if (m->empty())
    return false;

  FILE *fp = fopen(fname.c_str(), "w");
  if (!fp)
    return false;

  for(std::map<size_t, std::string>::iterator it = m->begin(); it != m->end(); it++) {
    //fprintf(fp, "%ld=%s\n", it->first, it->second.c_str());
    fprintf(fp, "%s\n", it->second.c_str());
  }

  fclose(fp);
  return true;
}

std::vector<std::unique_ptr<ComputeCapability>>
PHYnpuExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                      const IKernelLookup& /*kernel_lookup*/) const {
  // Find inputs, initializers and outputs for each supported subgraph
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::vector<std::string> unspportted_nodes;
  std::map<size_t, std::string> node_graphs;
  node_graphs.clear();

  // Handle If and Loop operators
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  auto model_name = graph_viewer.Name();
  auto model_path = graph_viewer.ModelPath();
  LOGS_DEFAULT(WARNING)<<"EP:phynpu get model_name:"<<model_name<<", model_path:"<<model_path;

  // Get EP:phynpu unspported nodes from the file: phynpu_unspported_nodes.txt
  std::string unspport_node_file = PHYNPU_UNSPPORTED_NODES_FILE;
  std::ifstream file(unspport_node_file);
  if(file.is_open()) {
    std::string lines;
    while (std::getline(file, lines)) {
      unspportted_nodes.push_back(lines);
    }
    std::remove(unspport_node_file.c_str());  
  }
  file.close();
  LOGS_DEFAULT(INFO)<<"EP:phynpu Unspported nodes like:";
  for(int i=0; i< unspportted_nodes.size(); i++) {
    LOGS_DEFAULT(INFO)<<unspportted_nodes[i];
  }

  // Need access to model_path_
  for (const auto& tensor : graph_viewer.GetAllInitializedTensors()) {
    if (tensor.second->has_data_location() &&
        tensor.second->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(INFO) << "PHYnpu: Initializers with external data"
                            " location are not currently supported";
      return result;
    }
  }

  // Get all the NodeUnits in the graph_viewer
  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const Node*, const NodeUnit*> node_unit_map;
  std::tie(node_unit_holder, node_unit_map) = QDQ::GetAllNodeUnits(graph_viewer);

  // This holds the result of whether a NodeUnit is supported or not,
  // to prevent nodes in a NodeUnit to be checked for multiple times
  std::unordered_map<const NodeUnit*, bool> node_unit_supported_result;
  node_unit_supported_result.reserve(node_unit_holder.size());
  std::unordered_set<std::string> node_outputs_in_current_group{};

  const auto is_node_supported = [&](const Node& node) -> bool {
    const NodeUnit* node_unit = node_unit_map.at(&node);
    bool supported = false;

    // If we have visited one of the nodes in the node_unit, use the result directly
    const auto it = node_unit_supported_result.find(node_unit);
    if (it != node_unit_supported_result.cend()) {
      supported = it->second;
    } else {
      // Check if the NodeUnit Name is unspportted_nodes txt
      auto it = std::find(unspportted_nodes.begin(), unspportted_nodes.end(), node_unit->Name());
      if(it != unspportted_nodes.end()) {
        supported = false;
      } else {
        // We only check the target node of the node unit
        supported = phytium_npu::GraphPhynpuEP::IsNodeSupportedInGroup(*node_unit, graph_viewer);
      }
      
      node_unit_supported_result[node_unit] = supported;
    }

    LOGS_DEFAULT(WARNING) << "Node supported: [" << supported
                          << "] Operator type: [" << node.OpType()
                          << "] index: [" << node.Index()
                          << "] name: [" << node.Name()
                          << "] as part of the NodeUnit type: [" << node_unit->OpType()
                          << "] index: [" << node_unit->Index()
                          << "] name: [" << node_unit->Name()
                          << "]";
    
    // record the node name for the node unit
    node_graphs[node.Index()] = node.Name();

    if (supported) {
      // We want to save all the output names of nodes in the current group for easy query
      for (const auto* output : node.OutputDefs()) {
        node_outputs_in_current_group.insert(output->Name());
      }
    }
    return supported;
  };

  const auto on_group_closed = [&](const std::vector<const Node*>& group) -> bool {
    // reset per-partition node group tracking
    node_outputs_in_current_group.clear();
    return true;
  };

  const auto gen_metadef_name = [&]() {
    static size_t group_counter = 0;
    return "PHYNPU_" + std::to_string(++group_counter);
  };

  result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported, on_group_closed,
                                            gen_metadef_name, PHYNPU, kPHYNPUExecutionProvider, &node_unit_map);

  std::for_each(result.begin(), result.end(), [&graph_viewer](auto& capability) {
    if (capability && capability->sub_graph && capability->sub_graph->GetMetaDef()) {
      const auto* meta_def = capability->sub_graph->GetMetaDef();
      bool has_any_non_constant_inputs = std::any_of(meta_def->inputs.begin(),
                                                     meta_def->inputs.end(), [&graph_viewer](const auto& input) {
                                                       return !graph_viewer.IsConstantInitializer(input, true);
                                                     });

      // ALL inputs are constant
      if (!has_any_non_constant_inputs) {
        capability.reset();
      }
    }
  });

  // save the node graph for each node unit into json file.
  if(node_graphs.size() > 0) {
    std::string filename = "graph_node.txt";
    WriteFile(filename, &node_graphs);
    LOGS_DEFAULT(INFO)<<"Record the graph nodes info into file: "<<filename;
  }
  
  const auto num_of_partitions = result.size();
  const auto num_of_supported_nodes = std::accumulate(
      result.begin(), result.end(), size_t{0},
      [](const auto& acc, const auto& partition) -> size_t {
        return acc + (partition && partition->sub_graph ? partition->sub_graph->nodes.size() : 0);
      });

  const auto summary_msg = MakeString(
      "PHYNPUExecutionProvider::GetCapability,",
      " number of partitions supported by PHYNPU: ", num_of_partitions,
      "; number of nodes in the graph: ", graph_viewer.NumberOfNodes(),
      "; number of nodes supported by PHYNPU: ", num_of_supported_nodes);

  // If the graph is partitioned in multiple subgraphs, and this may impact performance,
  // we want to give users a summary message at warning level.
  if (num_of_partitions > 1) {
    LOGS_DEFAULT(WARNING) << summary_msg;
  } else {
    LOGS_DEFAULT(WARNING) << summary_msg;
  }

  unspportted_nodes.clear();
  return result;
}


inline int64_t GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

int outCopyThread(phytium_npu::GraphPhynpuEP* graph_ep, int obj) {
  int nCount = (int)obj;
  {
    auto out_men = graph_ep->output_memory_[nCount].out_men;
    auto length  = graph_ep->output_memory_[nCount].length;
    auto device_ptr = graph_ep->LockMemory(out_men, PHYDNN_LOCK_ACCESS_READ_ONLY);
    memcpy(graph_ep->output_memory_[nCount].out_ptr, device_ptr, length);
    graph_ep->UnlockMemory(out_men);
  }
  return 0;
}

OrtValue CreateOrtValuebyPtr(void* ptr, TensorShape shape, ONNXTensorElementDataType type) {
  OrtValue ort_value;

  // 获取默认分配器（假设数据已经分配好）
  auto allocator = std::make_shared<CPUAllocator>();
  LOGS_DEFAULT(INFO) << "make tensor ptr by type:"<<type;
  std::unique_ptr<Tensor> tensor;

  // 创建 Tensor 对象
  if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<float>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<double>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<int32_t>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<int64_t>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<int16_t>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<int8_t>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<uint32_t>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<int64_t>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<uint16_t>(), shape, ptr, allocator);
  } else if(type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
    tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<uint8_t>(), shape, ptr, allocator);
  }
  
  // 初始化 OrtValue
  auto ml_tensor = DataTypeImpl::GetType<Tensor>();
  ort_value.Init(tensor.release(), ml_tensor, ml_tensor->GetDeleteFunc());
  return ort_value;
}

Status ExecuteStateFunc(phytium_npu::GraphPhynpuEP* graph_ep,
                        OrtKernelContext* context,
                        std::map<int, OrtValue>& input_id_map,
                        std::set<int>& input_args_vec) {
  Ort::KernelContext ctx(context);
  const size_t num_inputs = graph_ep->GetGraphInputs().size();
  LOGS_DEFAULT(INFO) << "ExecuteStateFunc input num_inputs:"<<num_inputs;

  std::map<uint32_t, ONNXTensorElementDataType> in_type_map;
  for (size_t i = 0; i < num_inputs; i++) {
    if (!graph_ep->GetGraphInputs()[i]->is_initializer) {
      const auto onnx_input_tensor = ctx.GetInput(i);
      const auto tensor_info = onnx_input_tensor.GetTensorTypeAndShapeInfo();
      auto type_size = onnxruntime::phytium_npu::GetTensorElementSize(tensor_info.GetElementType());
      auto in_shape = graph_ep->GetGraphInputs()[i]->shape.GetDims();
      auto dims = in_shape.size();
      auto length = graph_ep->GetGraphInputs()[i]->shape.SizeToDimension(dims) * type_size;
      in_type_map[i] = tensor_info.GetElementType();
      LOGS_DEFAULT(INFO) << "get input length:"<<length;

      auto& input_memory = graph_ep->input_memory_[i];
      if (!input_memory.first || input_memory.second < length) {
        if (input_memory.first) {
          graph_ep->DestroyMemory(input_memory.first);
        }
        input_memory.first = graph_ep->AllocateMemory(length);
        if(input_memory.first == nullptr) {
          LOGS_DEFAULT(ERROR) << "input AllocateMemory failed by the length:"<<length;
        }
        input_memory.second = length;
        graph_ep->AddBindingInput(graph_ep->input_info_[i], input_memory.first);
      }
      graph_ep->GetGraphInputs()[i]->is_initializer = true;
    }
  }

  // set input memory
  int in_index       = -1;
  int min_input_args = -1;
  for (size_t i = 0; i < num_inputs; i++) {
    bool bNodeInput = false;
    ctx.SetPhyInput(i, nullptr, &in_index);
    
    // 计算出第一次NPU子图运行时，图中输入的索引ID
    if(i == 0 && input_args_vec.size() >= num_inputs) {
      auto it = input_args_vec.begin();
      min_input_args = *it;
      if(in_index <= min_input_args) {
        std::advance(it, num_inputs-1);
        min_input_args = *it;
      }
    }
    LOGS_DEFAULT(INFO) <<"input i:"<<i<<", in_index:"<<in_index<<", min_input_args:"<<min_input_args;
    auto it = input_id_map.find(in_index);
    if(it != input_id_map.end()) { // found
      bNodeInput = true;
    }

    // 第一次推理时，需要记录下所有子图输入的索引ID和映射的张量,使用memcpy；
    // 从第二次推理开始，针对多个子图场景，第一子图必须直接memcpy拷贝数据，对第二个子图输入数据进行零拷贝
    if(!bNodeInput || in_index <= min_input_args) { // not found
      auto start2_time = GetCurrentUS();
      auto& input_memory = graph_ep->input_memory_[i]; 
      auto length = graph_ep->input_memory_[i].second;
    
      // 拷贝host数据到device，对于第一个子图，必须直接memcpy拷贝数据，无法进行零拷贝
      auto device_ptr = graph_ep->LockMemory(input_memory.first, PHYDNN_LOCK_ACCESS_WRITE_ONLY);     
      const auto onnx_input_tensor = ctx.GetInput(i);
      memcpy(device_ptr, onnx_input_tensor.GetTensorRawData(), length);
      void* ptr = const_cast<void*>(onnx_input_tensor.GetTensorRawData());
      TensorShape shape = graph_ep->GetGraphInputs()[i]->shape;
      OrtValue in_ort_value = CreateOrtValuebyPtr(device_ptr, shape, in_type_map[i]);
      graph_ep->UnlockMemory(input_memory.first);
      LOGS_DEFAULT(INFO) << "Input1 memcpy cost: " << GetCurrentUS() - start2_time << " us"<<" by size:"<<length/1024<<"KB"
                         <<", in index:"<<in_index<<", node_input_map.size:"<<input_id_map.size();

      // 针对多个子图场景，需要从第二次推理运行，仅在第一个子图中进行零拷贝
      if(bNodeInput) {
        for(auto pair=input_id_map.begin(); pair != input_id_map.end(); pair++) {
          auto start2_time = GetCurrentUS();
          int id_index = pair->first;
          int in_temp  = id_index;
          if(id_index > min_input_args) {
            ctx.SetPhyInput(i, &input_id_map[id_index], &in_temp);
            LOGS_DEFAULT(INFO) << "Input2 memcpy cost: " << GetCurrentUS() - start2_time << " us"<<", id_index:"<<id_index;
          }
        }
      }

      input_args_vec.insert(in_index);
      input_id_map.insert(std::make_pair(in_index, in_ort_value));
    } 
  }

  // get the output
  auto output_count = ctx.GetOutputCount();
  std::map<uint32_t, ONNXTensorElementDataType> out_type_map;
  LOGS_DEFAULT(INFO) << "ExecuteStateFunc output output_count:"<<output_count;
  for (size_t i = 0; i < output_count; i++) {
    auto out_shape = graph_ep->GetGraphOutputs()[i]->shape.GetDims();
    auto out_name  = graph_ep->GetGraphOutputs()[i]->name;
    auto dims      = out_shape.size();
    auto onnx_output_tensor =
        ctx.GetOutput(i, out_shape.data(), dims);
    for(int j=0; j<out_shape.size(); j++){
      LOGS_DEFAULT(INFO) << "output size:"<<out_shape[j]<<", index:"<<j; 
    }    
    const auto tensor_info = onnx_output_tensor.GetTensorTypeAndShapeInfo();
    auto type_size = onnxruntime::phytium_npu::GetTensorElementSize(tensor_info.GetElementType());
    auto length = graph_ep->GetGraphOutputs()[i]->shape.SizeToDimension(dims) * type_size;
    out_type_map[i] = tensor_info.GetElementType();
    LOGS_DEFAULT(INFO) << "get output length:"<<length<<", out_name:"<<out_name;

    auto& output_memory = graph_ep->output_memory_[i];
    if (!output_memory.out_men || output_memory.length < length) {
      if (output_memory.out_men) {
        graph_ep->DestroyMemory(output_memory.out_men);
      }
      output_memory.out_men = graph_ep->AllocateMemory(length);
      if(output_memory.out_men == nullptr) {
        LOGS_DEFAULT(ERROR) << "output AllocateMemory failed by the length:"<<length;
      }
      output_memory.length = length;
      graph_ep->AddBindingOutput(graph_ep->output_info_[i],
                                 output_memory.out_men);                
    }
  }

  // calc the times
  auto start_time = GetCurrentUS();
  graph_ep->ExecuteNetworkObject(true, 0, nullptr, nullptr);

  int out_index = 0;
  for (uint32_t k = 0; k < output_count; k++) {
    ctx.SetPhyOutput(k, nullptr, &out_index);
    auto it = graph_ep->node_output_map.find(out_index);
    
    // 对图中输出数据进行零拷贝，直接指针赋值，取消了memcpy拷贝操作
    if(it == graph_ep->node_output_map.end()) { // not found
      auto start2_time = GetCurrentUS();
      auto out_men   = graph_ep->output_memory_[k].out_men;
      auto length    = graph_ep->output_memory_[k].length;

      auto device_ptr = graph_ep->LockMemory(out_men, PHYDNN_LOCK_ACCESS_READ_ONLY);
      auto shape = graph_ep->GetGraphOutputs()[k]->shape;
      OrtValue out_ort_value = CreateOrtValuebyPtr(device_ptr, shape, out_type_map[k]);
      ctx.SetPhyOutput(k, &out_ort_value, &out_index);
      graph_ep->UnlockMemory(out_men);
      graph_ep->node_output_map[out_index] = out_ort_value;
      LOGS_DEFAULT(INFO) << "out1 memcpy cost: " << GetCurrentUS() - start2_time << " us"<<" by size:"<<length/1024<<"KB"<<", out_index:"<<out_index;
    } else {
      auto start2_time = GetCurrentUS();
      int index = 0;
      ctx.SetPhyOutput(k, &graph_ep->node_output_map[out_index], &index);
      LOGS_DEFAULT(INFO) << "out2 memcpy cost: " << GetCurrentUS() - start2_time << " us"<<", out index:"<<out_index;
    }
  }

  LOGS_DEFAULT(WARNING) << "Process cost: " << GetCurrentUS() - start_time << " us";
  return Status::OK();
}

common::Status PHYnpuExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
    std::vector<NodeComputeInfo>& node_compute_funcs) {  
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
    std::shared_ptr<phytium_npu::GraphPhynpuEP> graph_ep = std::make_shared<phytium_npu::GraphPhynpuEP>(graph_viewer);
   
    bool is_quant = false;
    auto input_count = graph_viewer.GetInputs().size();
    for (auto tensor : graph_viewer.GetInputs()) {
      LOGS_DEFAULT(INFO) << "subgraph input init:" << onnxruntime::phytium_npu::PrintNode(*tensor) << "#"
                            << graph_viewer.IsInitializedTensor(tensor->Name());
      auto input = std::make_shared<phytium_npu::GraphIOInfo>();
      input->name = tensor->Name();
      input->is_initializer = graph_viewer.IsConstantInitializer(tensor->Name(), true);
      graph_ep->GetGraphInputs().push_back(input);
      LOGS_DEFAULT(INFO) <<"input name:"<<input->name<<", is_initializer:"<<input->is_initializer;
    }

    auto output_count = graph_viewer.GetOutputs().size();
    for (auto tensor : graph_viewer.GetOutputs()) {
      LOGS_DEFAULT(INFO) << "subgraph output:" << onnxruntime::phytium_npu::PrintNode(*tensor);
      auto output = std::make_shared<phytium_npu::GraphIOInfo>();
      output->name = tensor->Name();
      output->is_initializer = false;
      graph_ep->GetGraphOutputs().push_back(output);
      LOGS_DEFAULT(INFO) <<"output name:"<<output->name<<", is_initializer:"<<output->is_initializer;
    }

    auto node_indices = graph_viewer.GetNodesInTopologicalOrder();
    int get_node_quant = 0;
    for (const auto& node_index : node_indices) {
      const auto node = graph_viewer.GetNode(node_index);
      const NodeUnit& node_unit = graph_ep->GetNodeUnit(node);

      // Only add op when we hit the target node
      if (node != &node_unit.GetNode()) {
        continue;
      }

      // get the first node to check if it is quantized
      if(get_node_quant == 0) {
        std::vector<NodeUnitIODef> input_defs = node_unit.Inputs();
        is_quant = input_defs[0].quant_param.has_value();
        get_node_quant += 1;
        LOGS_DEFAULT(INFO) << "get graph is quant:"<<is_quant;
      }

      LOGS_DEFAULT(WARNING) << "Adding node: [" << node->OpType() << "]";
      phytium_npu::SupportedBuiltinOps().at(node->OpType())->BuildOp(graph_ep.get(), graph_viewer, node_unit);
    }

    LOGS_DEFAULT(INFO) << "Verifying graph";
    {
      // Indentify the inputs and outputs
      auto input_count = graph_ep->GetGraphInputs().size();
      LOGS_DEFAULT(INFO) << "Model input count: " << input_count;
      std::vector<phydnn_tensor> input_tensors;
      if (input_count > 0) {
        input_tensors.resize(input_count);
        for (size_t i = 0; i < input_count; i++) {
          input_tensors[i] = *graph_ep->GetGraphInputs()[i]->tensor;
          if(input_tensors[i] == nullptr) {
            LOGS_DEFAULT(ERROR) << "get input_tensors failed.";
          }
        }
      }

      auto output_cnt = graph_ep->GetGraphOutputs().size();
      std::vector<phydnn_tensor> output_tensors;
      output_tensors.resize(output_cnt);
      for (size_t i = 0; i < output_cnt; i++) {
        output_tensors[i] = *graph_ep->GetGraphOutputs()[i]->tensor;
        if(output_tensors[i] == nullptr) {
          LOGS_DEFAULT(ERROR) << "get output_tensors failed.";
        }
      }
      graph_ep->CreateNetworkObject(input_tensors.size(),
                                    input_tensors.data(),
                                    output_tensors.size(),
                                    output_tensors.data(),
                                    is_quant);
      // Get the info of inputs and outputs, and check the count and buffer size of
      // inputs and outputs：
      graph_ep->output_memory_.clear();
      uint32_t num_inputs;
      graph_ep->GetNetworkObjectInputs(
          std::numeric_limits<unsigned int>::max(), nullptr, &num_inputs);
      if (num_inputs > 0) {
        graph_ep->input_info_.resize(num_inputs);
        graph_ep->input_memory_.resize(num_inputs,
                            std::pair<phydnn_memory, size_t>(nullptr, 0));
        graph_ep->GetNetworkObjectInputs(
            num_inputs, graph_ep->input_info_.data(), nullptr);
      }
      uint32_t num_outputs;
      graph_ep->GetNetworkObjectOutputs(
          std::numeric_limits<unsigned int>::max(), nullptr, &num_outputs);
      LOGS_DEFAULT(INFO) << "Verifying get num_inputs:"<<num_inputs<<",num_outputs:"<<num_outputs;

      graph_ep->output_info_.resize(num_outputs);
      graph_ep->output_memory_.resize(num_outputs);
      for(int i=0; i < num_outputs; i++) {
        graph_ep->output_memory_[i].out_men = nullptr;
        graph_ep->output_memory_[i].out_ptr = nullptr;
        graph_ep->output_memory_[i].length = 0;
      }
      graph_ep->GetNetworkObjectOutputs(
          output_cnt, graph_ep->output_info_.data(), nullptr);
      LOGS_DEFAULT(WARNING) << "Build success.";
    }

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [graph_ep](ComputeContext* /*context*/,
                                                FunctionState* state) {
      *state = graph_ep.get();
      return 0;
    };

    compute_info.compute_func =
        [graph_ep, this](FunctionState /*state*/, const OrtApi* /* api */,
                         OrtKernelContext* context) {
          std::lock_guard<OrtMutex> lock(this->GetMutex());
          Status res = ExecuteStateFunc(graph_ep.get(), context, input_id_map, input_args_vec);
          return res;
        };

    compute_info.release_state_func = [](FunctionState /*state*/) {};
    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> 
PHYnpuExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry 
       = std::make_shared<KernelRegistry>();
  return kernel_registry;
}

} // namespace onnxruntime
