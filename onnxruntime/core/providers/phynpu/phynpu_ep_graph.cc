 
 

#include <algorithm>
#include <dlfcn.h>
#include "core/providers/phynpu/phynpu_ep_graph.h"
#include "core/providers/phynpu/builders/op_builder_factory.h"
#include "core/framework/node_unit.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/optimizer/initializer.h"
#include "phynpu_util.h"

namespace onnxruntime {
namespace phytium_npu {

static void error_callback(phydnn_report_flags flags,
                           const char **tensor_names,
                           int num_tensor_names,
                           phydnn_err_code error_code,
                           const char *error_message) {
  std::string msg_prefix;
  switch (flags) {
    case phydnn_report_flags::PHYDNN_REPORT_ERROR:
      LOGS_DEFAULT(ERROR) << error_message;
      break;
    case phydnn_report_flags::PHYDNN_REPORT_VERBOSE:
      LOGS_DEFAULT(VERBOSE) << error_message;
      break;
    case phydnn_report_flags::PHYDNN_REPORT_INFO:
      LOGS_DEFAULT(INFO) << error_message;
      break;
    case phydnn_report_flags::PHYDNN_REPORT_WARNING:
      LOGS_DEFAULT(WARNING) << error_message;
      break;
    default:
      LOGS_DEFAULT(ERROR) << "Unknown report flag " << static_cast<int>(flags)
                          << " in error callback!";
  }
}

GraphPhynpuEP::GraphPhynpuEP(const onnxruntime::GraphViewer& graph_viewer) : graph_viewer_(graph_viewer) {
  Prepare();

// start phydnn
  phydnn_err_code error_code;
  phydnnSetErrorHandler(error_callback);
  graph_ = phydnnCreateNetwork(&error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnCreateNetwork!";
  }
  unsigned int num_devices = 0;
  auto err = phydnnGetDevices(PHYDNN_DEVICE_TYPE_ACCELERATOR, 1, &device_, &num_devices);
  if (err != PHYDNN_SUCCESS){
      LOGS_DEFAULT(ERROR) << " Could not get devices";
  }
  LOGS_DEFAULT(INFO) << "GraphPhynpuEP get devices num:" << num_devices;
  
  context_ = phydnnCreateContext(num_devices, &device_, 0, &error_code);
  if(context_){
    LOGS_DEFAULT(INFO) << "get context_";
  }
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnCreateContext!";
  }

  binding_ = phydnnCreateBinding(&error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnCreateBinding!";
  }
  if(binding_){
    LOGS_DEFAULT(INFO) << "get binding_";
  }
  node_output_map.clear();
  compiled_ = false;
}

GraphPhynpuEP::~GraphPhynpuEP() {
  for (size_t i = 0; i < input_memory_.size(); i++) {
    if (input_memory_[i].first) {
      DestroyMemory(input_memory_[i].first);
    }
  }
  for (size_t i = 0; i < output_memory_.size(); i++) {
    if (output_memory_[i].out_men) {
      DestroyMemory(output_memory_[i].out_men);
    }
  }

  input_info_.clear();
  tensors_.clear();
  graph_inputs_.clear();
  graph_outputs_.clear();
  output_info_.clear();
  input_memory_.clear();
  output_memory_.clear();
  buffers_.clear();
  node_output_map.clear();

  if (network_object_) {
    phydnnNetworkObjectDestroy(network_object_);
  }
  if (binding_) {
    phydnnBindingDestroy(binding_);
  }
  if (graph_) {
    phydnnNetworkDestroy(graph_);
  }
  if (context_) {
    phydnnContextDestroy(context_);
  }
}

bool GraphPhynpuEP::Prepare() {
  std::tie(node_unit_holder_, node_unit_map_) = QDQ::GetAllNodeUnits(graph_viewer_);
  return true;
}

const NodeUnit& GraphPhynpuEP::GetNodeUnit(const Node* node) const {
  const auto it = node_unit_map_.find(node);
  ORT_ENFORCE(it != node_unit_map_.end(), "Node does not have corresponding NodeUnit.");
  return *it->second;
}

bool GraphPhynpuEP::SupportedOp(const onnxruntime::GraphViewer& graph_viewer,
                          const NodeUnit& node_unit) {
  const auto& supported_builtins = phytium_npu::SupportedBuiltinOps();
  const auto& target_node = node_unit.GetNode();
  const auto& it = supported_builtins.find(target_node.OpType());
  if (supported_builtins.end() != it) {
    return it->second->IsSupported(graph_viewer, node_unit);
  }
  LOGS_DEFAULT(WARNING) << "Fallback unsupported op (node_unit) " << node_unit.OpType()
                        << " backward to devices like: cpu .";
  return false;
}

bool GraphPhynpuEP::IsNodeSupportedInGroup(const NodeUnit& node_unit, 
     const GraphViewer& graph_viewer) {
  return SupportedOp(graph_viewer, node_unit);
}

phydnn_tensor GraphPhynpuEP::CreateInputTensor(phydnn_tensor_descriptor *desc, bool is_qtensor) {
  phydnn_err_code error_code;
  if(is_qtensor && desc->type == PHYDNN_TYPE_I8) {
    desc->type = PHYDNN_TYPE_Q_I8;
  } else if(is_qtensor && desc->type == PHYDNN_TYPE_U8) {
    desc->type = PHYDNN_TYPE_Q_U8;
  }

  auto input_tensor = phydnnNetworkInput(graph_, desc, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkInput!";
  }
  return input_tensor;
}

uint8_t *GraphPhynpuEP::GetBuffer(size_t size) {
  buffers_.emplace_back(std::make_unique<uint8_t[]>(size));
  return buffers_.back().get();
}

phydnn_tensor GraphPhynpuEP::CreateFixedInputTensor(
    phydnn_tensor_descriptor *desc, const void *const data, bool copy) {
  phydnn_err_code error_code;
  phydnn_tensor fixed_input_tensor;

  if (copy) {
    auto size = phydnnGetDescriptorSize(desc, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnGetDescriptorSize!";
    }
    
    auto buffer = GetBuffer(size);
    memcpy(buffer, data, size);
    // LOGS_DEFAULT(INFO) << "CreateFixedInputTensor uint8 size:"<<size;
    // for(int i=0; i<20; i++) {
    //   LOGS_DEFAULT(INFO) << "data==" << static_cast<int>(buffer[i]);
    // }

    fixed_input_tensor =
      phydnnNetworkFixedInput(graph_, desc, buffer, &error_code);
  } else {
    LOGS_DEFAULT(INFO) << "CreateFixedInputTensor buffer data:";
    fixed_input_tensor =
      phydnnNetworkFixedInput(graph_, desc, data, &error_code);
  }
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
  }
  return fixed_input_tensor;
}


void GraphPhynpuEP::GetNetworkObjectInputs(unsigned int max_inputs,
                                           phydnn_input inputs[],
                                           unsigned int *num_inputs) {
  if(network_object_ == nullptr) {
    LOGS_DEFAULT(ERROR) << "network_object_ is nullptr!";
    return;
  }    
  phydnnNetworkObjectGetInputs(
      network_object_, max_inputs, inputs, num_inputs);
}

void GraphPhynpuEP::GetNetworkObjectOutputs(unsigned int max_outputs,
                                            phydnn_output outputs[],
                                            unsigned int *num_outputs) {
  if(network_object_ == nullptr) {
    LOGS_DEFAULT(ERROR) << "network_object_ is nullptr!";
    return;
  }   
  phydnnNetworkObjectGetOutputs(
      network_object_, max_outputs, outputs, num_outputs);
}

phydnn_tensor_descriptor GraphPhynpuEP::GetInputDescriptor(phydnn_input input) {
  phydnn_err_code error_code;
  phydnn_tensor_descriptor input_desc =
      phydnnGetInputDescriptor(input, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnGetInputDescriptor!";
  }    
  return input_desc;
}

phydnn_tensor_descriptor GraphPhynpuEP::GetOutputDescriptor(
    phydnn_output output) {
  phydnn_err_code error_code;
  phydnn_tensor_descriptor output_desc =
      phydnnGetOutputDescriptor(output, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnGetOutputDescriptor!";
  }       
  return output_desc;
}

phydnn_tensor_descriptor GraphPhynpuEP::GetTensorDescriptor(
    phydnn_tensor tensor) {
  phydnn_tensor_descriptor desc;
  phydnnGetTensorDescriptor(tensor, &desc);
  return desc;
}

size_t GraphPhynpuEP::GetDescriptorSize(
    const phydnn_tensor_descriptor *const descriptor) {
  phydnn_err_code error_code;
  auto size = phydnnGetDescriptorSize(descriptor, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnGetDescriptorSize!";
  }  
  return size;
}

void GraphPhynpuEP::AddBindingInput(phydnn_input input, phydnn_memory memory) {
  phydnnBindingAddInput(binding_, input, memory);
}

void GraphPhynpuEP::AddBindingOutput(phydnn_output output,
                                     phydnn_memory memory) {
  phydnnBindingAddOutput(binding_, output, memory);
}

phydnn_memory GraphPhynpuEP::AllocateMemory(size_t size) {
  phydnn_err_code error_code;
  auto memory = phydnnAllocateMemory(context_, size, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnImportMemory!";
  } 
  return memory;
}

void GraphPhynpuEP::DestroyMemory(phydnn_memory memory) {
  phydnnMemoryDestroy(memory);
}

void *GraphPhynpuEP::LockMemory(phydnn_memory memory,
                                phydnn_lock_access lock_access) {
  phydnn_err_code error_code;
  void *buffer = phydnnMemoryLock(memory, lock_access, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnMemoryLock!";
  }  
  return buffer;
}

void GraphPhynpuEP::UnlockMemory(phydnn_memory memory) {
  phydnnMemoryUnlock(memory);
}

bool GraphPhynpuEP::CheckConfigFileExists(const std::string &map_config_path) {
  if(access(map_config_path.c_str(), F_OK) != 0) {
    LOGS_DEFAULT(ERROR) << "Could not find or access Phytium NNA mapping config file:"<<map_config_path;
    return false;
  }   
  return true;
}

phydnn_network_object GraphPhynpuEP::CreateNetworkObject(
    unsigned int num_inputs,
    phydnn_tensor *inputs,
    unsigned int num_outputs,
    phydnn_tensor *outputs,
    bool is_quant) {
  phydnn_err_code error_code;
  const phydnn_network_object_flags flags = 0;
  // Get the directory of current module
  std::string cur_dir = ".";
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(error_callback), &dl_info);
  if (dl_info.dli_fname) {
    std::string dli_fname = dl_info.dli_fname;
    const size_t last_slash_idx = dli_fname.rfind('/');
    if (std::string::npos != last_slash_idx) {
      cur_dir = dli_fname.substr(0, last_slash_idx);
    }
  }

  const std::string in_josn_name = is_quant ? "/in8out8_d8_w8b8.json" : "/in32out32_d16_w16b16.json";
  const std::string map_config_path = cur_dir + in_josn_name;
  CheckConfigFileExists(map_config_path);
  std::string options;
  options += " -m " + map_config_path;
//  options += " -o ";
//  options += ".";
//  options += " --dump_debug_data ";
//  options += "enabled";
//  options += " --dump_debug_binaries ";
//  options += "enabled";
  // Add " --dump_debug_binaries enabled" to options if need debug info.
  network_object_ = phydnnCreateNetworkObject(device_,
                                              context_,
                                              graph_,
                                              num_inputs,
                                              inputs,
                                              num_outputs,
                                              outputs,
                                              flags,
                                              options.c_str(),
                                              &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnCreateNetworkObject!";
  }    
  return network_object_;
}

void GraphPhynpuEP::ExecuteNetworkObject(bool blocking_execute,
                                         unsigned int num_events_in_wait_list,
                                         const phydnn_event event_wait_list[],
                                         phydnn_event *event) {
  phydnnNetworkObjectExecute(network_object_,
                            binding_,
                            blocking_execute,
                            num_events_in_wait_list,
                            event_wait_list,
                            event);
}

void ConvertTophydnnDimensions(int32_t* input_dimensions,
                               uint32_t input_dimensions_count,
                               size_t* output_dimensions,
                               unsigned int* output_dimensions_count) {
  *output_dimensions_count = input_dimensions_count;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    auto dimension = input_dimensions[i];
    if(dimension < 0) {
      LOGS_DEFAULT(ERROR) << "check the dimension:"<<dimension<<" < 0";
    }
    output_dimensions[i] = dimension;
  }
}

template <typename T>
phydnn_tensor GraphPhynpuEP::QuantizeLinear(phydnn_tensor_descriptor desc,
                                           float scale,
                                           int32_t zero_point,
                                           const std::vector<T>& input) {
  desc.quant_param.scale      = scale;
  desc.quant_param.zero_point = zero_point;

  std::vector<uint8_t> quantized_data;
  quantized_data.resize(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    // 根据量化公式计算量化值
    float quantized_value = input[i] / scale + zero_point;
    
    // 四舍五入到最近的整数
    quantized_value = std::round(quantized_value);

    // 饱和处理
    quantized_value = std::clamp(quantized_value, 
                           static_cast<float>(std::numeric_limits<int8_t>::min()), 
                           static_cast<float>(std::numeric_limits<int8_t>::max()));

    // 存储结果
    quantized_data[i] = static_cast<int8_t>(quantized_value);
  }
  const void* buffer = static_cast<const void*>(quantized_data.data());
  auto tensor = CreateFixedInputTensor(&desc, buffer, true); 
  return tensor;
}

template <typename T>
phydnn_tensor GraphPhynpuEP::QuantizeLinear(phydnn_tensor_descriptor desc, 
                                            const std::vector<T>& input) {
  std::vector<int8_t> output(input.size());
  // 计算输入数据的最小值和最大值
  T min_val = std::numeric_limits<T>::max();
  T max_val = -std::numeric_limits<T>::max();
  for (const auto& val : input) {
    if (val < min_val) min_val = val;
    if (val > max_val) max_val = val;
  }

  // 计算量化尺度 (scale) 和零点 (zero_point)
  // 假设量化到int8范围 [-128, 127]
  const int8_t quant_min = -128;
  const int8_t quant_max = 127;
  
  // 假设量化到uint8范围 [0, 255]
  // const uint8_t quant_min = 0;
  // const uint8_t quant_max = 255;

  T range = max_val - min_val;
  float scale = range / (quant_max - quant_min);  // scale = (max - min) / (quant_max - quant_min)
  int8_t zero_point = static_cast<int8_t>(std::round(quant_min - min_val / scale));

  // 确保zero_point在有效范围内
  zero_point = std::max(quant_min, std::min(quant_max, zero_point));
  for (size_t i = 0; i < input.size(); ++i) {
    // 计算量化值
    T quantized_value = input[i] / scale + zero_point;
    
    // 饱和操作，确保结果在int8范围内
    int8_t saturated_value = std::max<int8_t>(quant_min, std::min<int8_t>(quant_max, static_cast<int8_t>(std::round(quantized_value))));
    output[i] = saturated_value;
  }

  LOGS_DEFAULT(INFO) << "QuantizeLinear scale:"<<scale<<", zero_point:"<<zero_point;
  desc.quant_param.scale      = scale;
  desc.quant_param.zero_point = zero_point;

  const void* buffer = static_cast<const void*>(output.data());
  auto tensor = CreateFixedInputTensor(&desc, buffer, true); 
  return tensor;
}


phydnn_tensor GraphPhynpuEP::CreatePhynpuTensor(phydnn_tensor_descriptor desc,
                             const ONNX_NAMESPACE::TensorProto* tensor_proto,
                             ONNX_NAMESPACE::TensorProto_DataType data_type,
                             size_t size,
                             bool is_qtensor) {
  phydnn_tensor tensor;
  phydnn_err_code error_code;
  auto shape_tensor = *tensor_proto;
  Initializer unpacked_tensor(shape_tensor);
  LOGS_DEFAULT(INFO) << "CreatePhynpuTensor data_type:"<<data_type<<", size:"<<size<<", is_qtensor:"<<is_qtensor;

  if(is_qtensor) {
    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
      desc.type  = PHYDNN_TYPE_Q_I8;
      auto data = unpacked_tensor.DataAsSpan<int8_t>();
      const void* buffer = static_cast<const void*>(data.data());
      tensor = CreateFixedInputTensor(&desc, buffer, true); 
      return tensor;
    } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
      desc.type  = PHYDNN_TYPE_Q_U8;
      auto data = unpacked_tensor.DataAsSpan<uint8_t>();
      const void* buffer = static_cast<const void*>(data.data());
      tensor = CreateFixedInputTensor(&desc, buffer, true);
      return tensor;
    } 
  }

  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    auto raw_shape = unpacked_tensor.DataAsSpan<int64_t>(); 
    const auto size = SafeInt<size_t>(shape_tensor.dims()[0]);
    std::vector<int32_t> raw_data(size);
    for (uint32_t i = 0; i < size; i++) {
      int32_t data = SafeInt<int32_t>(raw_shape[i]);
      raw_data[i] = data;
    }
    const void* buffer = static_cast<const void*>(raw_data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true);
    return tensor;
  } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_UINT64) {
    auto raw_shape = unpacked_tensor.DataAsSpan<uint64_t>(); 
    const auto size = SafeInt<size_t>(shape_tensor.dims()[0]);
    std::vector<int32_t> raw_data(size);
    for (uint32_t i = 0; i < size; i++) {
      int32_t data = SafeInt<int32_t>(raw_shape[i]);
      raw_data[i] = data;
    }
    const void* buffer = static_cast<const void*>(raw_data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true);
    return tensor;
  }

  if (desc.type == PHYDNN_TYPE_U8) {
    auto data = unpacked_tensor.DataAsSpan<uint8_t>(); 
    const void* buffer = static_cast<const void*>(data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true);
  } else if (desc.type == PHYDNN_TYPE_I8) {
    auto data = unpacked_tensor.DataAsSpan<int8_t>(); 
    const void* buffer = static_cast<const void*>(data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true); 
  } else if (desc.type == PHYDNN_TYPE_I16) {
    auto data = unpacked_tensor.DataAsSpan<int16_t>(); 
    const void* buffer = static_cast<const void*>(data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true); 
  } else if (desc.type == PHYDNN_TYPE_U16) {
    auto data = unpacked_tensor.DataAsSpan<uint16_t>(); 
    const void* buffer = static_cast<const void*>(data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true); 
  } else if (desc.type == PHYDNN_TYPE_F16) {
    auto data = unpacked_tensor.DataAsSpan<float>(); // 此次未定义：float16
    const void* buffer = static_cast<const void*>(data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true); 
  } else if (desc.type == PHYDNN_TYPE_F32) {
    auto data = unpacked_tensor.DataAsSpan<float>(); 
    const void* buffer = static_cast<const void*>(data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true); 
  } else if (desc.type == PHYDNN_TYPE_I32) {
    auto data = unpacked_tensor.DataAsSpan<int32_t>(); 
    const void* buffer = static_cast<const void*>(data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true); 
  } else if (desc.type == PHYDNN_TYPE_U32) {
    auto data = unpacked_tensor.DataAsSpan<uint32_t>(); 
    const void* buffer = static_cast<const void*>(data.data());
    tensor = CreateFixedInputTensor(&desc, buffer, true); 
  } else {
    LOGS_DEFAULT(ERROR) << "CreatePhynpuTensor unsupported tensor type:"<<desc.type;
  }

  return tensor;          
}

std::shared_ptr<phydnn_tensor> GraphPhynpuEP::UpdateIntputTensor(
    const onnxruntime::GraphViewer& graph_viewer, const NodeUnitIODef nudef, const NodeUnit& node_unit) {
  const auto& arg = nudef.node_arg;

  if (tensors_.end() != tensors_.find(arg.Name())) {
    return tensors_.find(arg.Name())->second;
  }

  if(arg.Name() == "") {
    LOGS_DEFAULT(ERROR) << "UpdateIntputTensor arg.Name() is empty";
    return std::make_shared<phydnn_tensor>(nullptr);
  }

  bool is_constant  = graph_viewer.IsConstantInitializer(arg.Name(), true);
  bool is_qtensor   = nudef.quant_param.has_value();

  float scale = 0.0f;
  int32_t zp  = 0;
  if(is_qtensor) {
    std::optional<std::vector<float>> scales;
    std::optional<std::vector<int32_t>> zps;
    GetQuantizationScaleAndZeroPoint(graph_viewer,
                                    nudef, node_unit.ModelPath(),
                                    scale, zp, scales, zps);
    LOGS_DEFAULT(INFO)<<"UpdateIntputTensor qtensor scale:"<<scale<<", zp:"<<zp<<", scales is:"<<scales.has_value()<<", zps is:"<<zps.has_value();                                      
  }
  LOGS_DEFAULT(INFO)<<"UpdateIntputTensor create tensors by arg.Name():"<<arg.Name()<<", is_qtensor:"<<is_qtensor<<", is_constant:"<<is_constant;

  // get dimensions
  auto shape_proto = arg.Shape();
  std::vector<int32_t> dimensions;
  if (shape_proto != nullptr) {
    for (int i = 0; i < shape_proto->dim_size(); i++) {
      auto dim = shape_proto->dim(i);
      dimensions.push_back(dim.dim_value());
    }
  }
  if (dimensions.size() == 0) {
    dimensions.push_back(1);
  }

  phydnn_tensor tensor;
  phydnn_tensor_descriptor desc;
  ConvertTophydnnDimensions(
      dimensions.data(), dimensions.size(), desc.size, &desc.dimensions);
  
  auto type_proto = arg.TypeAsProto();
  ONNX_NAMESPACE::TensorProto_DataType data_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  if (type_proto->has_tensor_type()) {
    data_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type_proto->tensor_type().elem_type());
    desc.type = GetPhyDnnType(data_type, is_qtensor);  
    if(is_qtensor) {
      desc.quant_param.scale      = scale;
      desc.quant_param.zero_point = zp;
    } 
    LOGS_DEFAULT(INFO) << "type_proto data_type:"<<data_type<<", desc.type:"<<desc.type;
  }

  phydnn_err_code error_code;
  size_t length = phydnnGetDescriptorSize(&desc, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnGetDescriptorSize!";
  }

  // constant
  if(is_constant) {
    if(desc.dimensions == 1 && desc.size[0] == 0) { // 输入常量为空，比如resize算子的输入
      return std::make_shared<phydnn_tensor>(nullptr);
    } else {
      const ONNX_NAMESPACE::TensorProto* tensor_proto =
        graph_viewer.GetConstantInitializer(arg.Name(), true);
      if(tensor_proto != nullptr) {
        tensor = CreatePhynpuTensor(desc, tensor_proto, data_type, length, is_qtensor);
      } else {
        tensor = CreateInputTensor(&desc, is_qtensor);
      }
    }
  } else {
    tensor = CreateInputTensor(&desc, is_qtensor);
  }

  // input tensors
  for (auto& input : graph_inputs_) {
    if (input->name == arg.Name()) {
      input->tensor = std::make_shared<phydnn_tensor>(tensor);
      input->shape = GetTensorShape(arg);
      break;
    }
  }

  tensors_.insert({arg.Name(), std::make_shared<phydnn_tensor>(tensor)});
  return std::make_shared<phydnn_tensor>(tensor);
}

std::shared_ptr<phydnn_tensor> GraphPhynpuEP::UpdateOutputTensor(
    const NodeUnitIODef nudef, phydnn_tensor tensor) {
  const auto& arg = nudef.node_arg;
  if (tensors_.end() != tensors_.find(arg.Name())) {
    return tensors_.find(arg.Name())->second;
  }

  LOGS_DEFAULT(INFO) << "UpdateOutputTensor Name:"<<arg.Name();
  // output tensors
  for (auto& output : graph_outputs_) {
    if (output->name == arg.Name()) {
      output->tensor = std::make_shared<phydnn_tensor>(tensor);
      output->shape = utils::GetTensorShapeFromTensorShapeProto(*arg.Shape());
      if(output->tensor == nullptr) {
        LOGS_DEFAULT(ERROR) << "UpdateOutputTensor set output tensor failed.";
      }
      LOGS_DEFAULT(INFO) << "set OutputTensor Name:"<<arg.Name()<<",output->shape:"<<output->shape.NumDimensions();
      break;
    }
  }

  tensors_.insert({arg.Name(), std::make_shared<phydnn_tensor>(tensor)});
  return std::make_shared<phydnn_tensor>(tensor);
}

}  // namespace phytium_npu
}  // namespace onnxruntime

