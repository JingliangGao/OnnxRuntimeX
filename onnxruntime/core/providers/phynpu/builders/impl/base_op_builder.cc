 
 

#include <string>
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

// onnx版本opset支持范围 [1 ~ 22]
bool BaseOpBuilder::HasSupportedOpSet(const NodeUnit& node_unit) const {
  auto since_version = node_unit.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node_unit) 
      || since_version > GetMaxSupportedOpSet(node_unit)) {
    LOGS_DEFAULT(WARNING) << node_unit.OpType() << " opset [" << since_version
                          << "] is only supported for opset ["
                          << GetMinSupportedOpSet(node_unit) << ", "
                          << GetMaxSupportedOpSet(node_unit) << "]";
    return false;
  }
  return true;
}

// 判断图中哪些节点是支持的
bool BaseOpBuilder::IsSupported(const onnxruntime::GraphViewer& graph_viewer,
                                const NodeUnit& node_unit) const {
  auto initializers = graph_viewer.GetAllInitializedTensors();
  if (!HasSupportedOpSet(node_unit)) {
    return false;
  }
  if (!HasSupportedInputOutputs(initializers, node_unit)) {
    return false;
  }
  return IsOpSupported(graph_viewer, &node_unit.GetNode());
}

// 检查一个计算图节点的输入和输出数据类型是否被支持
bool IsTypeSupported(const NodeArg* node_arg) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  switch (type_proto->tensor_type().elem_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
      return true;
    default:
      return false;
  }
}

bool BaseOpBuilder::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit) const {
  // Check input/output data type, int64 is generally unsupported
  // specific op builder can override this if the int64 input corresponds to PHYNPU param
  // for (const auto& input : node_unit.Inputs()) {
  //   auto input_type = input.node_arg.Type();
  //   if (*input_type == "tensor(int64)" || !IsTypeSupported(&input.node_arg)) {
  //     LOGS_DEFAULT(INFO) << node_unit.OpType() << " has unsupported input type : "
  //                           << *input_type;
  //     return false;
  //   }
  // }
  
  for (const auto& output : node_unit.Outputs()) {
    auto output_type = output.node_arg.Type();
    if (*output_type == "tensor(int64)" || !IsTypeSupported(&output.node_arg)) {
      LOGS_DEFAULT(WARNING) << node_unit.OpType() << " has unsupported output type : "
                            << *output_type;
      return false;
    }
  }
  return true;
}

// 节点常规和量化检查
bool BaseOpBuilder::HasSupportedInputOutputs(const InitializedTensorSet& initializers,
                                             const NodeUnit& node_unit) const {
  // We do not support unknown(null) input shape
  auto has_supported_shape = [](const NodeArg& node_arg, const std::string& name, const std::string& op_type) {
    const auto* shape_proto = node_arg.Shape();
    if (!shape_proto) {
      LOGS_DEFAULT(WARNING) << "Node [" << name << "] type [" << op_type
                            << "] Input [" << node_arg.Name() << "] has no shape";
      return false;
    }

    // We do not support dynamic shape input yet, but resize op's second input can be empty
    for (const auto& dim : shape_proto->dim()) {
      if (!dim.has_dim_value()) {
        LOGS_DEFAULT(WARNING) << "Dynamic shape is not supported for now, for input:" << node_arg.Name();
        return false;
      }
      if (dim.dim_value() == 0 && op_type != "Resize") {
        LOGS_DEFAULT(WARNING) << "Zero in shape is not supported for now, for input:" << node_arg.Name();
        return false;
      }
    }
    return true;
  };

  auto has_initialized_quant_param = [](const NodeArg& arg, const InitializedTensorSet& initializers) {
    auto it = initializers.find(arg.Name());
    if (it == initializers.end()) {
      LOGS_DEFAULT(WARNING) << "The quantization param must be an initializer tensor";
      return false;
    }
    return true;
  };

  for (const auto& input : node_unit.Inputs()) {
    if (!input.node_arg.Exists()) {
      continue;
    }
    if (!has_supported_shape(input.node_arg, node_unit.Name(), node_unit.OpType()))
      return false;

    if (input.quant_param.has_value()) {
      if (!has_supported_shape(input.quant_param->scale, node_unit.Name(), node_unit.OpType()))
        return false;

      if (!has_initialized_quant_param(input.quant_param->scale, initializers))
        return false;
      // zero point is optional
      if (input.quant_param->zero_point) {
        if (!has_supported_shape(*input.quant_param->zero_point, node_unit.Name(), node_unit.OpType()))
          return false;
        if (!has_initialized_quant_param(*input.quant_param->zero_point, initializers))
          return false;
        if (input.quant_param->zero_point->Type() != input.node_arg.Type()) {
          LOGS_DEFAULT(ERROR) << "Invalid input type because the data type mismatch with its' quant param type.";
          return false;
        }
      }
    }
  }
  for (const auto& output : node_unit.Outputs()) {
    for (const auto& dim : output.node_arg.Shape()->dim()) {
      if (!dim.has_dim_value()) {
        LOGS_DEFAULT(WARNING) << "Dynamic shape is not supported for now, for output:" << output.node_arg.Name();
        return false;
      }
      if (dim.dim_value() == 0 && output.node_arg.Shape()->dim_size() > 1) {
        LOGS_DEFAULT(WARNING) << "Zero in shape is not supported for now, for output:" << output.node_arg.Name();
        return false;
      }
    }
    if (output.quant_param.has_value()) {
      if (!has_supported_shape(output.quant_param->scale, node_unit.Name(), node_unit.OpType()))
        return false;

      if (!has_initialized_quant_param(output.quant_param->scale, initializers))
        return false;
      // zero point is optional
      if (output.quant_param->zero_point) {
        if (!has_supported_shape(*output.quant_param->zero_point, node_unit.Name(), node_unit.OpType()))
          return false;
        if (!has_initialized_quant_param(*output.quant_param->zero_point, initializers))
          return false;
      }
    }
  }
  return HasSupportedInputOutputsImpl(initializers, node_unit);
}

// 创建BaseOpBuilder实例
bool BaseOpBuilder::BuildOp(phytium_npu::GraphPhynpuEP* graph_ep,
                            const onnxruntime::GraphViewer& graph_viewer,
                            const NodeUnit& node_unit) {
  std::vector<std::shared_ptr<phydnn_tensor>> inputs;
  std::vector<NodeUnitIODef> input_defs  = node_unit.Inputs();
  std::vector<NodeUnitIODef> output_defs = node_unit.Outputs();
  bool is_lut = GetLUTOpsname(node_unit);
  LOGS_DEFAULT(WARNING)<<"BuildOp node unit name:"<<node_unit.Name()<<", optype:"<<node_unit.OpType()<<", is_lut:"<<is_lut;   

  // get quantization param by th first input and output tensor
  phydnn_quant_param in_quant, out_quant;
  float scale = 0.0f;
  int32_t zp  = 0;
  std::optional<std::vector<float>> scales;
  std::optional<std::vector<int32_t>> zps;

  const auto& arg_in  = input_defs[0].node_arg;   
  const auto& arg_out = output_defs[0].node_arg;  
  bool is_qtensor     = input_defs[0].quant_param.has_value();  
  bool is_qtensor_out = output_defs[0].quant_param.has_value();  
  if (is_qtensor) {
    GetQuantizationScaleAndZeroPoint(graph_viewer,
                                     input_defs[0], node_unit.ModelPath(),
                                     scale, zp, scales, zps);
    in_quant.scale = scale;
    in_quant.zero_point = zp;              
    LOGS_DEFAULT(INFO)<<"BuildOp1 name:"<<arg_in.Name()<<", input scale:"<<scale<<", zp:"<<zp;   
  }

  if(is_qtensor_out) {
    GetQuantizationScaleAndZeroPoint(graph_viewer,
                                     output_defs[0], node_unit.ModelPath(),
                                     scale, zp, scales, zps);
    out_quant.scale = scale;
    out_quant.zero_point = zp;                                                      
    LOGS_DEFAULT(INFO)<<"BuildOp2 name:"<<arg_out.Name()<<", output scale:"<<scale<<", zp:"<<zp;       
  }

// get input tensor
  for (const auto input_def : input_defs) {
    auto tensor = graph_ep->UpdateIntputTensor(graph_viewer, input_def, node_unit);
    inputs.push_back(tensor);
  }

// HandleBuildOp
  auto output_tensor = HandleBuildOp(graph_ep, inputs, node_unit, in_quant, out_quant);

// get output tensor
  if(output_tensor.size() > 0 && output_tensor.size() == output_defs.size()) {
    int index = 0;
    for (auto output_def : output_defs) {
      // set quant output tensor
      auto out_tesnsor = SetQuantOutputTensor(graph_viewer, graph_ep, output_def, 
                                              node_unit, output_tensor[index++], is_lut);
      graph_ep->UpdateOutputTensor(output_def, out_tesnsor);
    }
  } else {
    LOGS_DEFAULT(ERROR) <<"Failed! BuildOp get output_tensors size:"<<output_tensor.size()<<", but required output size:"<<output_defs.size();
    return false;
  }
  return true;
}

}  // namespace phytium_npu
}  // namespace onnxruntime
