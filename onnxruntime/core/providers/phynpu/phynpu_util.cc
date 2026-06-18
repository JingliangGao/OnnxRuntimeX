 
 

 
#include <algorithm>
#include <cassert>
#include <cstdint>
#include "core/providers/shared/utils/utils.h"
#include "phynpu_util.h"

namespace onnxruntime {
namespace phytium_npu {
  
template <typename T>
struct shared_array_deletor {
  void operator()(T const* ptr) { delete[] ptr; }
};

phydnn_type GetPhyDnnType(ONNX_NAMESPACE::TensorProto_DataType to_type, bool is_qtensor) {
  phydnn_type operand_type;
  if(is_qtensor) {
    if(to_type == ONNX_NAMESPACE::TensorProto_DataType_INT8){
      operand_type = PHYDNN_TYPE_Q_I8;
      return operand_type;
    } else if(to_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8){
      operand_type = PHYDNN_TYPE_Q_U8;
      return operand_type;
    }
  }

  switch (to_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        operand_type = PHYDNN_TYPE_U8;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        operand_type = PHYDNN_TYPE_I8;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
        operand_type = PHYDNN_TYPE_I16;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
        operand_type = PHYDNN_TYPE_U16;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        operand_type = PHYDNN_TYPE_F16;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        operand_type = PHYDNN_TYPE_F32;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        operand_type = PHYDNN_TYPE_I32;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
        operand_type = PHYDNN_TYPE_U32;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        operand_type = PHYDNN_TYPE_I32;
        break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
        operand_type = PHYDNN_TYPE_U32;
       break;
    default:
        LOGS_DEFAULT(ERROR) <<"The Cast node has unsupported 'to' type, name: "<< to_type;
  }

  return operand_type;
}

std::string PrintNode(const onnxruntime::NodeArg& node_arg) {
  auto shape = node_arg.Shape();
  if (shape == nullptr) {
    return "<null>";
  }
  std::string s = node_arg.Name() + ":<";
  if (shape->dim_size() == 0) {
    s += "1>, is a scalar";
    return s;
  }
  for (int i = 0; i < shape->dim_size(); i++) {
    auto dim = shape->dim(i);
    std::string s1;
    std::stringstream ss;
    ss << dim.dim_value();
    ss >> s1;
    s += s1;
    if (i < shape->dim_size() - 1) {
      s += ",";
    } else {
      s += ">";
    }
  }
  return s;
}

std::string PrintNode(const std::vector<int64_t> shape) {
  if (shape.size() == 0) {
    return "<null>";
  }
  std::string s = "<";
  for (std::size_t i = 0; i < shape.size(); i++) {
    auto dim = shape[i];
    std::string s1;
    std::stringstream ss;
    ss << dim;
    ss >> s1;
    s += s1;
    if (i < shape.size() - 1) {
      s += ",";
    } else {
      s += ">";
    }
  }
  return s;
}

TensorShape GetTensorShape(const onnxruntime::NodeArg& node_arg) {
    auto shape_proto = node_arg.Shape();
    std::vector<int64_t> dims;
    if (shape_proto != nullptr) {
        for (int i = 0; i < shape_proto->dim_size(); i++) {
        auto dim = shape_proto->dim(i);
        dims.push_back(dim.dim_value());
        }
    }
    if (dims.size() == 0) {
        dims.push_back(1);
    }
    TensorShape ts(dims);
    return ts;
}

void UpdateConv2DPadAndDilation(
    int32_t input_size,
    int32_t filter_height_or_width,
    std::string auto_pad,
    int32_t* pad_top_or_left,
    int32_t* pad_bottom_or_right,
    int32_t stride_height_or_width,
    int32_t* dilation_height_or_width) {
  if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    auto output_size =
        (input_size + stride_height_or_width - 1) / stride_height_or_width;
    auto pad_size = (output_size - 1) * stride_height_or_width +
                    filter_height_or_width - input_size;
    pad_size = pad_size < 0 ? 0 : pad_size;
    *pad_top_or_left = pad_size / 2;
    *pad_bottom_or_right = pad_size - *pad_top_or_left;
    *dilation_height_or_width = 1;
  } else if (auto_pad == "VALID") {
    *pad_top_or_left = 0;
    *pad_bottom_or_right = 0;
  }
}

std::shared_ptr<uint8_t> UnpackTensor(
    const NodeArg* node_arg, const ONNX_NAMESPACE::TensorProto& initializer) {
  std::shared_ptr<uint8_t> unpackedTensor;
  auto shape = GetTensorShape(*node_arg);
  size_t elementCount = shape.Size();
  LOGS_DEFAULT(INFO) << "UnpackTensor data_type:"<<initializer.data_type();

#define CASE_PROTO(X, Y)                                                      \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X: {      \
    size_t tensorByteSize = elementCount * sizeof(Y);                         \
    unpackedTensor.reset(new uint8_t[tensorByteSize],                         \
                         shared_array_deletor<uint8_t>());                    \
    auto status = onnxruntime::utils::UnpackTensor(                           \
        initializer,                                                          \
        initializer.has_raw_data() ? initializer.raw_data().data() : nullptr, \
        initializer.has_raw_data() ? initializer.raw_data().size() : 0,       \
        reinterpret_cast<Y*>(unpackedTensor.get()), elementCount);            \
    if (!status.IsOK()) {                                                     \
      LOGS_DEFAULT(ERROR) << "Unpack tensor data failed.";                    \
    }                                                                         \
    break;                                                                    \
  }
  switch (initializer.data_type()) {
    CASE_PROTO(FLOAT, float);
    CASE_PROTO(DOUBLE, double);
    CASE_PROTO(BOOL, bool);
    CASE_PROTO(INT8, int8_t);
    CASE_PROTO(INT16, int16_t);
    CASE_PROTO(INT32, int32_t);
    CASE_PROTO(INT64, int64_t);
    CASE_PROTO(UINT64, uint64_t);
    CASE_PROTO(UINT8, uint8_t);
    CASE_PROTO(UINT16, uint16_t);
    CASE_PROTO(UINT32, uint32_t);
    CASE_PROTO(FLOAT16, onnxruntime::MLFloat16);
    default:
      return nullptr;
  }

  return unpackedTensor;
}

std::vector<phydnn_tensor> CreateElementwiseOps(
    phytium_npu::GraphPhynpuEP* graph_ep,
    phydnn_tensor input0_tensor,
    phydnn_tensor input1_tensor,
    phydnn_operation_binary phydnn_operation,
    int32_t fuse_code) {
  phydnn_err_code error_code;
  phydnn_tensor_descriptor input0_desc;
  phydnn_tensor_descriptor input1_desc;
  std::vector<phydnn_tensor> output_tensors;

  phydnnGetTensorDescriptor(input0_tensor, &input0_desc);
  phydnnGetTensorDescriptor(input1_tensor, &input1_desc);
  auto input0_type = input0_desc.type;
  auto input1_type = input1_desc.type;
  LOGS_DEFAULT(INFO)<<"CreateElementwiseOps input0_desc.type:"<<input0_desc.type
                       <<", input1_type.type:"<<input1_desc.type;

  for(unsigned int i=0; i<input0_desc.dimensions; i++) {
   LOGS_DEFAULT(INFO)<<"CreateElementwiseOps input0_desc.size:"<<input0_desc.size[i];
  }

  for(unsigned int i=0; i<input1_desc.dimensions; i++) {
   LOGS_DEFAULT(INFO)<<"CreateElementwiseOps input1_desc.size:"<<input1_desc.size[i];
  }
  
  if (input0_type != input1_type) {
    LOGS_DEFAULT(INFO)<<"set input1_tensor type:"<<input1_desc.type;
    input1_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                input1_tensor,
                                input0_desc.type,
                                &input0_desc.quant_param,
                                &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call CreateElementwiseOps, error_code:" << error_code;
      return output_tensors;
    } 
  }

  auto output_tensor = phydnnNetworkBinaryOp(
      graph_ep->GetGraph(), input0_tensor, input1_tensor, phydnn_operation, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code;
    return output_tensors;
  } 
 
  phydnn_tensor_descriptor output_desc;
  phydnnGetTensorDescriptor(output_tensor, &output_desc);
  for(unsigned int i=0; i<output_desc.dimensions; i++){
    LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
  }

  // Add activation
  if (fuse_code == 1) { // relu
    output_tensor = phydnnNetworkReLUOp(graph_ep->GetGraph(),
                                        output_tensor,
                                        true, 0.0, false, 0.0, false, 
                                        &error_code);
    if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkReLUOp, error_code:" << error_code;
    return output_tensors;
    }
  } else if (fuse_code == 2) { // relu1
    output_tensor = phydnnNetworkReLUOp(graph_ep->GetGraph(),
                                        output_tensor,
                                        true, 0.0, true, 1.0, false, 
                                        &error_code);
    if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkReLU1Op, error_code:" << error_code;
    return output_tensors;
    }
  } else if (fuse_code == 3) { // relu6
    output_tensor = phydnnNetworkReLUOp(graph_ep->GetGraph(),
                                        output_tensor,
                                        true, 0.0, true, 6.0, false, 
                                        &error_code);
    if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkReLU6Op, error_code:" << error_code;
    return output_tensors;
    }
  } else {
    LOGS_DEFAULT(ERROR) << "ElementwiseOps Unsupported fuse_code:"<< fuse_code;
  }

  output_tensors.push_back(output_tensor);
  return output_tensors;  
}


std::vector<phydnn_tensor> CreateReLULayer(phytium_npu::GraphPhynpuEP* graph_ep,
                            phydnn_tensor input_tensor,
                            bool has_min_clamp,
                            float min_clamp,
                            bool has_max_clamp,
                            float max_clamp,
                            float negative_slope) {
  std::vector<phydnn_tensor> output_tensors;                            
  phydnn_err_code error_code;
  phydnn_tensor_descriptor input_desc, output_desc;
  auto output_tensor = phydnnNetworkReLUOp(graph_ep->GetGraph(),
                                          input_tensor,
                                          has_min_clamp,
                                          min_clamp,
                                          has_max_clamp,
                                          max_clamp,
                                          negative_slope,
                                          &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReLUOp, error_code:" << error_code;
    return output_tensors;
  } 
  
  phydnnGetTensorDescriptor(input_tensor, &input_desc);
  phydnnGetTensorDescriptor(output_tensor, &output_desc);
  LOGS_DEFAULT(INFO)<<"CreateReLULayer input_desc type:"<<input_desc.type<<", output_desc type:"<<output_desc.type;

  if (output_desc.type != input_desc.type) {
      output_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                      output_tensor,
                                      input_desc.type,
                                      &input_desc.quant_param,
                                      &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
      return output_tensors;
    } 
  }

  phydnnGetTensorDescriptor(output_tensor, &output_desc);
  for(unsigned int i=0; i<output_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"get result: "<< i <<" output_desc.size:"<<output_desc.size[i];
  }

  output_tensors.push_back(output_tensor);
  return output_tensors; 
}

std::vector<phydnn_tensor> CreateConvertUnaryActivations(phytium_npu::GraphPhynpuEP* graph_ep,
                                        phydnn_tensor input_tensor,
                                        phydnn_operation_unary op) {
  std::vector<phydnn_tensor> output_tensors;
  phydnn_err_code error_code;
  phydnn_tensor_descriptor input_desc;
  phydnnGetTensorDescriptor(input_tensor, &input_desc);
  LOGS_DEFAULT(INFO) <<"CreateConvertUnaryActivations input_desc.type"<<input_desc.type;

  auto output_tensor = 
       phydnnNetworkUnaryOp(graph_ep->GetGraph(), input_tensor, op, &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "Failed to call CreateConvertUnaryActivations, error_code:" << error_code;
    return output_tensors;
  } 

  phydnn_tensor_descriptor output_desc;
  phydnnGetTensorDescriptor(output_tensor, &output_desc);
  for(unsigned int i=0; i<output_desc.dimensions; i++){
    LOGS_DEFAULT(INFO) <<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
  }

  output_tensors.push_back(output_tensor);
  return output_tensors;
}

size_t GetTensorElementSize(const ONNXTensorElementDataType type) {
  switch (type) {
    case onnx::TensorProto_DataType_INT64:
    case onnx::TensorProto_DataType_UINT64:
    case onnx::TensorProto_DataType_DOUBLE:
      return 8;
    case onnx::TensorProto_DataType_FLOAT:
    case onnx::TensorProto_DataType_INT32:
    case onnx::TensorProto_DataType_UINT32:
      return 4;
    case onnx::TensorProto_DataType_FLOAT16:
    case onnx::TensorProto_DataType_INT16:
    case onnx::TensorProto_DataType_UINT16:
      return 2;
    case onnx::TensorProto_DataType_INT8:
    case onnx::TensorProto_DataType_UINT8:
    case onnx::TensorProto_DataType_BOOL:
      return 1;
    default:
      break;
  }
  return 0;
}

size_t GetTensorBytes(const Ort::TensorTypeAndShapeInfo& info) {
  return info.GetElementCount() * GetTensorElementSize(info.GetElementType());
}

std::vector<phydnn_tensor> PoolBaseOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_pooling_type pool_type) {
  std::vector<phydnn_tensor> output_tensors;            
  NodeAttrHelper helper(node_unit.GetNode());
  auto ksize   = helper.Get("kernel_shape", std::vector<uint32_t>{1U, 1U});
  auto strides = helper.Get("strides", std::vector<uint32_t>{1U, 1U});
  auto pads    = helper.Get("pads", std::vector<uint32_t>{0U, 0U, 0U, 0U});
  auto ceil_mode = helper.Get("ceil_mode", 0);  
  bool count_include_pad = (bool)helper.Get("count_include_pad", 0);

  unsigned int size[2]         = {ksize[0], ksize[1]};
  unsigned int stride[2]       = {strides[0], strides[1]};
  unsigned int pad_begin[2]    = {pads[0], pads[1]};
  unsigned int pad_end[2]      = {pads[2], pads[3]};

  phydnn_tensor_descriptor desc;
  phydnn_err_code error_code;
  auto input_tensor = *inputs[0];
  phydnnGetTensorDescriptor(input_tensor, &desc);
  LOGS_DEFAULT(INFO)<<"PoolOpBuilder input_desc.type:"<<desc.type;
  for(unsigned int i=0; i<desc.dimensions; i++){
    LOGS_DEFAULT(INFO) <<"input index:"<<i<<" input_desc.size:"<<desc.size[i];
  }

  // Adjust padding to be compatible with ceil_mode
  unsigned int pool_h, pool_w;
  if (ceil_mode == 0) {
    pool_h = ((desc.size[2] + pad_begin[0] + pad_end[0] - size[0]) /
        stride[0]) + 1;
    pool_w = ((desc.size[3] + pad_begin[1] + pad_end[1] - size[1]) /
        stride[1]) + 1;
  } else {
    pool_h = ((desc.size[2] + pad_begin[0] + pad_end[0] - size[0] +
        stride[0] - 1) / stride[0]) + 1;
    pool_w = ((desc.size[3] + pad_begin[1] + pad_end[1] - size[1] +
        stride[1] - 1) / stride[1]) + 1;
  }
  if ((pool_h - 1) * stride[0] + size[0] >= desc.size[2] + pad_begin[0])
    pad_end[0] = (pool_h - 1) * stride[0] + size[0] - desc.size[2] - pad_begin[0];

  if ((pool_w - 1) * stride[1] + size[1] >= desc.size[3] + pad_begin[1])
    pad_end[1] = (pool_w - 1) * stride[1] + size[1] - desc.size[3] - pad_begin[1];

  auto output_tensor = phydnnNetworkPooling2dOp_v3(graph_ep->GetGraph(),
                                                  input_tensor,
                                                  size,
                                                  stride,
                                                  pad_begin,
                                                  pad_end,
                                                  pool_type,
                                                  count_include_pad,
                                                  &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "PoolOpBuilder failed, error_code:" << error_code;
    return output_tensors;
  } 

  phydnn_tensor_descriptor output_desc;
  phydnnGetTensorDescriptor(output_tensor, &output_desc);
  for(unsigned int i=0; i<output_desc.dimensions; i++){
    LOGS_DEFAULT(INFO) <<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
  }

  output_tensors.push_back(output_tensor);
  return output_tensors;                                    
}

std::vector<phydnn_tensor> GolabelPoolBaseOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_pooling_type pool_type) {
  std::vector<phydnn_tensor> output_tensors;            
  phydnn_tensor_descriptor desc;
  phydnn_err_code error_code;
  auto input_tensor = *inputs[0];
  phydnnGetTensorDescriptor(input_tensor, &desc);
  LOGS_DEFAULT(INFO)<<"GolabelPoolBaseOp input_desc.type:"<<desc.type;

  bool count_include_pad = false;
  unsigned int ceil_mode = 0;            
  unsigned int size[2]         = {desc.size[2], desc.size[3]};
  unsigned int stride[2]       = {1, 1};
  unsigned int pad_begin[2]    = {0, 0};
  unsigned int pad_end[2]      = {0, 0};
  
  // Adjust padding to be compatible with ceil_mode
  unsigned int pool_h, pool_w;
  if (ceil_mode == 0) {
    pool_h = ((desc.size[2] + pad_begin[0] + pad_end[0] - size[0]) /
        stride[0]) + 1;
    pool_w = ((desc.size[3] + pad_begin[1] + pad_end[1] - size[1]) /
        stride[1]) + 1;
  } else {
    pool_h = ((desc.size[2] + pad_begin[0] + pad_end[0] - size[0] +
        stride[0] - 1) / stride[0]) + 1;
    pool_w = ((desc.size[3] + pad_begin[1] + pad_end[1] - size[1] +
        stride[1] - 1) / stride[1]) + 1;
  }
  if ((pool_h - 1) * stride[0] + size[0] >= desc.size[2] + pad_begin[0])
    pad_end[0] = (pool_h - 1) * stride[0] + size[0] - desc.size[2] - pad_begin[0];

  if ((pool_w - 1) * stride[1] + size[1] >= desc.size[3] + pad_begin[1])
    pad_end[1] = (pool_w - 1) * stride[1] + size[1] - desc.size[3] - pad_begin[1];

  auto output_tensor = phydnnNetworkPooling2dOp_v3(graph_ep->GetGraph(),
                                                  input_tensor,
                                                  size,
                                                  stride,
                                                  pad_begin,
                                                  pad_end,
                                                  pool_type,
                                                  count_include_pad,
                                                  &error_code);
  if(error_code != PHYDNN_SUCCESS) {
    LOGS_DEFAULT(ERROR) << "PoolOpBuilder failed, error_code:" << error_code;
    return output_tensors;
  } 

  phydnn_tensor_descriptor output_desc;
  phydnnGetTensorDescriptor(output_tensor, &output_desc);
  for(unsigned int i=0; i<output_desc.dimensions; i++){
    LOGS_DEFAULT(INFO) <<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
  }
  
  output_tensors.push_back(output_tensor);
  return output_tensors;                                      
}

std::vector<phydnn_tensor> BinaryBaseBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                       
                      const NodeUnit& node_unit,
                      phydnn_operation_binary op_type) {                                      
  LOGS_DEFAULT(INFO) << "Creating BinaryOp by type:"<<op_type;   
  auto input0_tensor = *inputs[0];                                                            
  auto input1_tensor = *inputs[1];       
  std::vector<phydnn_tensor> output_tensors;    
  phydnn_err_code error_code;                                                     
  phydnn_tensor_descriptor input0_desc, input1_desc;                                          
  phydnnGetTensorDescriptor(input0_tensor, &input0_desc);                                     
  phydnnGetTensorDescriptor(input1_tensor, &input1_desc);                                                        
  for(unsigned int i=0; i<input0_desc.dimensions; i++){                                         
    LOGS_DEFAULT(INFO)<<"BinaryOp input_desc0.size:"<<input0_desc.size[i];                 
  }                                                                                           
  for(unsigned int i=0; i<input1_desc.dimensions; i++){                                                          
    LOGS_DEFAULT(INFO)<<"BinaryOp input1_desc.size:"<<input1_desc.size[i];                 
  }        

  if(input0_desc.type != input1_desc.type) {
    LOGS_DEFAULT(INFO)<<"BinaryOp input0 type:"<<input0_desc.type<<", input1 type:"<<input1_desc.type;
    input1_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                input1_tensor,
                                input0_desc.type,
                                &input0_desc.quant_param,
                                &error_code);
    if(error_code != PHYDNN_SUCCESS) {                                                          
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code; 
      return output_tensors;                                                                             
    }
  }
                                                                                     
  auto output_tensor = phydnnNetworkBinaryOp(                                                 
        graph_ep->GetGraph(), input0_tensor, input1_tensor, op_type, &error_code);      
  if(error_code != PHYDNN_SUCCESS) {                                                          
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code; 
    return output_tensors;                                                                             
  }   
  
  output_tensors.push_back(output_tensor);
  return output_tensors;                                                                          
}  

phydnn_tensor MMBinaryBuildOp(phytium_npu::GraphPhynpuEP* graph_ep,                             
                      phydnn_tensor x_tensor,
                      phydnn_tensor y_tensor,  
                      phydnn_operation_binary op_type,                  
                      const NodeUnit& node_unit) {                                      
  LOGS_DEFAULT(INFO) << "Creating MM BinaryOp by PHYDNN_OPERATION_MATMUL";   
  phydnn_err_code error_code;                                                  
  phydnn_tensor_descriptor x_desc, y_desc;                                          
  phydnnGetTensorDescriptor(x_tensor, &x_desc);                                     
  phydnnGetTensorDescriptor(y_tensor, &y_desc);                                                        
  for(unsigned int i=0; i<x_desc.dimensions; i++){                                         
    LOGS_DEFAULT(INFO)<<"BinaryOp input_desc0.size:"<<x_desc.size[i];                 
  }                                                                                           
  for(unsigned int i=0; i<y_desc.dimensions; i++){                                                          
    LOGS_DEFAULT(INFO)<<"BinaryOp input1_desc.size:"<<y_desc.size[i];                 
  }                                                                                                                    

  if(x_desc.dimensions < y_desc.dimensions){
    for(int i=0; i<y_desc.dimensions-x_desc.dimensions; i++) {
      x_tensor = phydnnNetworkBroadcastOp(graph_ep->GetGraph(),x_tensor,0,y_desc.size[y_desc.dimensions-x_desc.dimensions-1-i],&error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBroadcastOp, error_code:" << error_code; 
        return nullptr;    
      }
    }
  } else if(x_desc.dimensions > y_desc.dimensions){
    for(int i=0; i<x_desc.dimensions-y_desc.dimensions; i++) {
      y_tensor = phydnnNetworkBroadcastOp(graph_ep->GetGraph(),y_tensor,0,x_desc.size[x_desc.dimensions-y_desc.dimensions-1-i],&error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBroadcastOp, error_code:" << error_code; 
        return nullptr;    
      }
    }
  }

  if(x_desc.type != y_desc.type) {
    y_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                y_tensor,
                                x_desc.type,
                                &x_desc.quant_param,
                                &error_code);
    if(error_code != PHYDNN_SUCCESS) {                                                          
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code; 
      return nullptr;                                                                             
    }
  }

  auto output_tensor = phydnnNetworkBinaryOp(                                                 
        graph_ep->GetGraph(), x_tensor, y_tensor, op_type, &error_code);      
  if(error_code != PHYDNN_SUCCESS) {                                                          
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp, error_code:" << error_code; 
    return nullptr;                                                                             
  }                                                                                           
  return output_tensor;                                                                       
}  

std::vector<phydnn_tensor> ReduceBaseBuildOp(                                                                    
          phytium_npu::GraphPhynpuEP* graph_ep,                                                       
          std::vector<std::shared_ptr<phydnn_tensor>>& inputs,                                  
          const NodeUnit& node_unit,
          phydnn_reduce_type reduce_type) {                                                 
  phydnn_tensor_descriptor input_desc;  
  phydnn_tensor_descriptor output_desc;                                                          
  phydnn_err_code error_code;                                                                         
  auto input_tensor = *inputs[0];  
  std::vector<phydnn_tensor> output_tensors;    
  bool is_quant = false;                                                         
  phydnnGetTensorDescriptor(input_tensor, &input_desc);                                         
  LOGS_DEFAULT(INFO)<<"ReduceOpBuilder input_desc.type:"<<input_desc.type<<" by reduce_type:"<<reduce_type;       
  if(input_desc.type >= PHYDNN_TYPE_Q_I8) {
    is_quant = true;
  }

  // set axes
  NodeAttrHelper helper(node_unit.GetNode());                                                   
  std::vector<int32_t> def_axes;
  if(reduce_type == PHYDNN_REDUCE_ARGMAX) {
    auto axis = helper.Get("axis", 0);
    def_axes.push_back(axis);
  } else {
    if (node_unit.SinceVersion() < 18 && helper.HasAttr("axes")) { // ??since opset 18, axes is required
      def_axes = helper.Get("axes", def_axes);
    } else if (inputs.size() > 1) {
      const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
      const auto& axes_tensor = *initializers.at(node_unit.GetNode().InputDefs()[1]->Name());
      Initializer unpacked_tensor(axes_tensor);
      auto raw_axes = unpacked_tensor.DataAsSpan<int64_t>();
      for (uint32_t i = 0; i < raw_axes.size(); i++) {
        int32_t data = SafeInt<int32_t>(raw_axes[i]);
        def_axes.push_back(data);
      }
    } else {
      for (int32_t i = 0; i < input_desc.dimensions; i++) {
        def_axes.push_back(i);
      }
    }  
  }

  std::vector<int32_t> axes(def_axes.begin(), def_axes.end());
  for(int i=0; i<axes.size(); i++) {
    if(axes[i] < 0) {                                                                                
      axes[i] += input_desc.dimensions;                                                                
    }    
    LOGS_DEFAULT(INFO)<<"ReduceOpBuilder axes size:"<<axes.size()<<" index:"<<i<<" axes:"<<axes[i];      
  }

  bool keepdim = helper.Get("keepdims", 1) == 1;  
  const auto to_type = helper.Get("dtype", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);         
  phydnn_type operand_type;  
  if(is_quant && to_type == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    operand_type = PHYDNN_TYPE_Q_I8;
  } else if(is_quant && to_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    operand_type = PHYDNN_TYPE_Q_U8;
  } else{                                                                
    switch (to_type) {                                                                            
      case ONNX_NAMESPACE::TensorProto_DataType_BOOL:                                             
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:                                            
        operand_type = PHYDNN_TYPE_U8;                                                            
        break;                                                                                             
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:                                             
        operand_type = PHYDNN_TYPE_I8;                                                                    
        break;                                                                                    
      case ONNX_NAMESPACE::TensorProto_DataType_INT16:                                            
        operand_type = PHYDNN_TYPE_I16;                                                           
        break;                                                                                       
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16:                                           
        operand_type = PHYDNN_TYPE_U16;                                                           
        break;                                                                                    
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:                                          
        operand_type = PHYDNN_TYPE_F16;                                                           
        break;                                                                                    
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:                                            
        operand_type = PHYDNN_TYPE_F32;                                                                                         
        break;                                                                                    
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:                                            
        operand_type = PHYDNN_TYPE_I32;                                                                                                 
        break;                                                                                                                       
      case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      case ONNX_NAMESPACE::TensorProto_DataType_UINT64:                                           
        operand_type = PHYDNN_TYPE_U32;                                                           
        break;                                                                                        
      default:                                                                                    
          LOGS_DEFAULT(ERROR) <<"The Cast node has unsupported 'to' type, name: "<< to_type;      
    }   
  }  
                                                                                                                                                         
  phydnn_tensor output_tensor = phydnnNetworkReduceOp(graph_ep->GetGraph(),                   
                                                  input_tensor,                               
                                                  reduce_type,                             
                                                  axes.data(),                                            
                                                  axes.size(),                                
                                                  &error_code);                               
  if(error_code != PHYDNN_SUCCESS) {                                                          
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkReduceOp, error_code:" << error_code; 
    return output_tensors;                                                                           
  }                                                                                                                                   
  if(operand_type != input_desc.type){                                                           
    output_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(), output_tensor, operand_type, nullptr, &error_code); 
    if(error_code != PHYDNN_SUCCESS) {                                                        
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code; 
      return output_tensors;                                                                                                
    }                                                                                         
  }      

  phydnnGetTensorDescriptor(output_tensor, &output_desc);                                                                                                                       
  if(keepdim) {                                                                               
    phydnn_tensor_descriptor new_desc;                                                        
    new_desc.dimensions = input_desc.dimensions;                                              
    new_desc.type       = output_desc.type;                                                      
    for (int i = 0; i < (int)input_desc.dimensions; i++) {                                   
      if (std::find(axes.begin(), axes.end(), i) == axes.end()){                              
        new_desc.size[i] = input_desc.size[i];                                                
      } else{                                                                                 
        new_desc.size[i] = 1;                                                                     
      }                                                                                       
    }  
    LOGS_DEFAULT(INFO)<<"ReduceOpBuilder new_desc.dimensions: "<<new_desc.dimensions<<" new_desc.type:"<<new_desc.type;                                                                                        
    output_tensor = phydnnNetworkReshapeOp(graph_ep->GetGraph(), output_tensor, &new_desc, &error_code); 
    if(error_code != PHYDNN_SUCCESS) {                                                        
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkReshapeOp, error_code:" << error_code;  
      return output_tensors;                                                                                                               
    }                                                                                         
  }                                                                                            
                                                              
  phydnnGetTensorDescriptor(output_tensor, &output_desc);                                         
  for(unsigned int i=0; i<output_desc.dimensions; i++) {                                      
    LOGS_DEFAULT(INFO)<<"ReduceOpBuilder get result: "<<i<<" output_desc.size:"<<output_desc.size[i]; 
  }  

  output_tensors.push_back(output_tensor);
  return output_tensors;                                                                         
}              


bool GetType(const NodeArg& node_arg, int32_t& type) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS_DEFAULT(INFO) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool GetClipMinMaxImpl(std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)> get_const_initializer,
                       const Node& node, float& min, float& max) {
  const auto& node_name = node.Name();
  int32_t input_type;
  min = std::numeric_limits<float>::lowest();
  max = std::numeric_limits<float>::max();
  if (!GetType(*node.InputDefs()[0], input_type)) {
    return false;
  }

  if (node.SinceVersion() < 11) {  // Clip opset 1, 6 is using attributes for min/max
    NodeAttrHelper helper(node);
    // attributes will be always float
    min = helper.Get("min", std::numeric_limits<float>::lowest());
    max = helper.Get("max", std::numeric_limits<float>::max());
  } else {
    auto get_value =
        [&](const ONNX_NAMESPACE::TensorProto* initializer, std::string_view type, float& value) -> bool {
      if (!initializer) {
        LOGS_DEFAULT(INFO) << " input of Clip must be a constant initializer"<< type;
        return false;
      }

      Initializer unpacked_tensor_min(*initializer);
      switch (input_type) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          value = unpacked_tensor_min.DataAsSpan<float>()[0];
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
          value = unpacked_tensor_min.DataAsSpan<MLFloat16>()[0].ToFloat();
          break;
        default:
          LOGS_DEFAULT(INFO) << "GetClipMinMax() only supports float and float16 as min and max inputs for now."
                                << " The node [" << node_name << "] has input type: " << input_type;
          return false;
      }

      return true;
    };

    // min and max are both optional. could have neither, one or both.
    if (node.InputDefs().size() > 1 && node.InputDefs()[1]->Exists()) {
      // we have input min
      const auto& min_name = node.InputDefs()[1]->Name();
      const auto* min_value = get_const_initializer(min_name);
      if (!get_value(min_value, "Min", min)) {
        return false;
      }
    }

    if (node.InputDefs().size() > 2 && node.InputDefs()[2]->Exists()) {
      // we have input max
      const auto& max_name = node.InputDefs()[2]->Name();
      const auto* max_value = get_const_initializer(max_name);
      if (!get_value(max_value, "Max", max)) {
        return false;
      }
    }
  }

  return true;
}

bool GetphynpuClipMinMax(const GraphViewer& graph_viewer, const Node& node, float& min, float& max) {
  return GetClipMinMaxImpl(
      [&graph_viewer](const std::string& name) -> const ONNX_NAMESPACE::TensorProto* {
        return graph_viewer.GetConstantInitializer(name);
      },
      node, min, max);
}

std::vector<int32_t> GetDataFromInputTensors(phytium_npu::GraphPhynpuEP* graph_ep, 
                            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
                            const NodeUnit& node_unit,
                            int index) {                   
  const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
  const auto& shape_tensor = *initializers.at(node_unit.GetNode().InputDefs()[index]->Name());
  const auto& type_tensor  = node_unit.GetNode().InputDefs()[index]->TypeAsProto()->tensor_type().elem_type();

  Initializer unpacked_tensor(shape_tensor);
  if (type_tensor == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64) {
    auto raw_shape = unpacked_tensor.DataAsSpan<int64_t>();
    const auto size = SafeInt<size_t>(shape_tensor.dims()[0]);
    std::vector<int32_t> raw_data(size);
    for (uint32_t i = 0; i < size; i++) {
      int32_t data = SafeInt<int32_t>(raw_shape[i]);
      raw_data[i] = data;
    }
    return raw_data;
  } else {
    auto raw_shape = unpacked_tensor.DataAsSpan<int32_t>();
    const auto size = SafeInt<size_t>(shape_tensor.dims()[0]);
    std::vector<int32_t> raw_data(size);
    for (uint32_t i = 0; i < size; i++) {
      int32_t data = SafeInt<int32_t>(raw_shape[i]);
      raw_data[i] = data;
    }
    return raw_data;
  }
}

void GetQuantizationScaleAndZeroPoint(
    const GraphViewer& graph_viewer, 
    const NodeUnitIODef& io_def, 
    const std::filesystem::path& model_path,
    float& scale, 
    int32_t& zero_point, 
    std::optional<std::vector<float>>& pcq_scales,
    std::optional<std::vector<int32_t>>& pcq_zps) {
 
  scale = 0.0f;
  zero_point = 0;

  const auto& quant_param = *io_def.quant_param;
  {  // get the scale
    const auto& name = quant_param.scale.Name();
    const auto* s = graph_viewer.GetConstantInitializer(name);
    if (!s) {
      LOGS_DEFAULT(ERROR) << name + " is not a constant initializer";
    };
    Initializer unpacked_tensor(*s, model_path);
    scale = unpacked_tensor.DataAsSpan<float>()[0];

    // per channel quantized handling
    if (!unpacked_tensor.dims().empty() && unpacked_tensor.dims()[0] != 0 && unpacked_tensor.dims()[0] != 1) {
      auto scales = unpacked_tensor.DataAsSpan<float>();
      std::vector<float> scales_vec(scales.begin(), scales.end());
      pcq_scales = onnxruntime::make_optional(std::move(scales_vec));
    }
  }

  if (quant_param.zero_point) {  // get the zero point if it exists
    const auto& name = quant_param.zero_point->Name();
    const auto* s = graph_viewer.GetConstantInitializer(name);
    if (!s) {
      LOGS_DEFAULT(ERROR) << name + " is not a constant initializer";
    };
    Initializer unpacked_tensor(*s, model_path);
    bool is_i8_zp = unpacked_tensor.data_type() == onnx::TensorProto_DataType_INT8;
    // some qdq conv bias is int32 quantized
    bool is_int32_zp = unpacked_tensor.data_type() == onnx::TensorProto_DataType_INT32;
    zero_point = is_i8_zp ? static_cast<int32_t>(unpacked_tensor.DataAsSpan<int8_t>()[0])
                 : is_int32_zp ? static_cast<int32_t>(unpacked_tensor.DataAsSpan<int32_t>()[0])
                               : static_cast<int32_t>(unpacked_tensor.DataAsByteSpan()[0]);

    // per channel quantized handling
    if (!unpacked_tensor.dims().empty() && unpacked_tensor.dims()[0] != 0 && unpacked_tensor.dims()[0] != 1) {
      auto type = unpacked_tensor.data_type();
      if (is_i8_zp) {
        auto zps = unpacked_tensor.DataAsSpan<int8_t>();
        std::vector<int32_t> zps_vec(zps.begin(), zps.end());
        pcq_zps = onnxruntime::make_optional(std::move(zps_vec));
      } else if (is_int32_zp) {
        auto zps = unpacked_tensor.DataAsByteSpan();
        std::vector<int32_t> zps_vec(zps.begin(), zps.end());
        pcq_zps = onnxruntime::make_optional(std::move(zps_vec));
      } else {
        auto zps = unpacked_tensor.DataAsSpan<int32_t>();
        std::vector<int32_t> zps_vec(zps.begin(), zps.end());
        pcq_zps = onnxruntime::make_optional(std::move(zps_vec));
      }
    }
  }
}

phydnn_tensor ConvertQuantTensorType(phytium_npu::GraphPhynpuEP* graph_ep,
                                    phydnn_tensor tensor, 
                                    phydnn_type out_type,
                                    phydnn_quant_param *quant_param) {
  phydnn_type type;
  if (out_type == PHYDNN_TYPE_Q_I8 || out_type == PHYDNN_TYPE_I8) {
    type = PHYDNN_TYPE_Q_I8;
  } else {
    type = PHYDNN_TYPE_Q_U8;
  }

  LOGS_DEFAULT(INFO)<<"Convert QuantTensorType type:"<<type<<", scaler:"<<quant_param->scale<<", zp:"<<quant_param->zero_point;
  phydnn_err_code error_code;
  auto quant_tensor =
      phydnnNetworkCastOp(graph_ep->GetGraph(), tensor, type, quant_param, &error_code);
  if(error_code != PHYDNN_SUCCESS) {                                                        
    LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkReshapeOp, error_code:" << error_code;  
    return nullptr;                                                                                                               
  } 
  return quant_tensor;
}

phydnn_tensor SetQuantOutputTensor(const onnxruntime::GraphViewer& graph_viewer,
                                  phytium_npu::GraphPhynpuEP* graph_ep,
                                  const NodeUnitIODef nudef,
                                  const NodeUnit& node_unit,
                                  phydnn_tensor tensor,
                                  bool is_lut) {
  const auto& arg  = nudef.node_arg;
  bool is_qtensor  = nudef.quant_param.has_value();
  LOGS_DEFAULT(INFO)<<"Set Out tensor arg.Name():"<<arg.Name()<<", is_qtensor:"<<is_qtensor<<", is_lut:"<<is_lut;

  //quant && LUT查表算子的输出不需要在量化了
  if(is_qtensor && !is_lut) {
    // get output quantization params
    float scale = 0.0f;
    int32_t zp  = 0;
    std::optional<std::vector<float>> scales;
    std::optional<std::vector<int32_t>> zps;

    auto type_proto  = arg.TypeAsProto();
    auto data_type   = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type_proto->tensor_type().elem_type());
    auto type        = GetPhyDnnType(data_type, is_qtensor); 

    GetQuantizationScaleAndZeroPoint(graph_viewer,
                                    nudef, 
                                    node_unit.ModelPath(),
                                    scale, zp, 
                                    scales, zps);
    phydnn_quant_param quant_param;
    quant_param.scale      = scale;
    quant_param.zero_point = zp;                                 
    LOGS_DEFAULT(INFO)<<"Out BuildOp type:"<<type<<", scale:"<<scale<<", zp:"<<zp<<", scales is:"<<scales.has_value();  

    auto output_tensors = ConvertQuantTensorType(graph_ep, tensor, type, &quant_param);
    return output_tensors;
  }
  
  return tensor;
}

// 激活算子量化时，是否为查表操作
bool GetLUTOpsname(const NodeUnit& node_unit) {
  bool is_lut = false;
  if( node_unit.OpType() == "HardSigmoid" || 
      node_unit.OpType() == "HardSwish"   || 
      node_unit.OpType() == "Sigmoid"     || 
      node_unit.OpType() == "LeakyRelu"   || 
      node_unit.OpType() == "Tanh"        ||
      node_unit.OpType() == "Exp"         ) {
    is_lut = true;
  }
  return is_lut;
}

// HardSigmoid 函数
float hard_sigmoid(float x, float alpha, float beta) {
  return std::min(std::max(0.0f, alpha * x + beta), 1.0f);
}

// HardSwish 函数
float hardswish(float x) {
  const float relu6 = std::min(std::max(0.0f, x + 3.0f), 6.0f);
  return x * relu6 / 6.0f;
}

// 生成HardSigmoid查找表
std::vector<int32_t> GenerateQuantizedHardSigmoidLUT(phydnn_quant_param& out_attrs, 
                                                     phydnn_quant_param& in_attrs, 
                                                     size_t lut_size,
                                                     bool is_signed,
                                                     float alpha,
                                                     float beta) {
  std::vector<int32_t> lut(lut_size); // 假设输入是8位整数，所以大小为256

  for (int i = 0; i < lut_size; ++i) {
    int index = is_signed ? i - 128 : i;

    // Dequantize: 根据输入的scale和zero_point还原原始值
    float val = (index - in_attrs.zero_point) * in_attrs.scale;
    float output_data;
    
    // Apply the hard sigmoid function
    // float result = std::min(std::max(0.0f, val + 3.0f), 6.0f) / 6.0f; // HardSwish
    float result = hard_sigmoid(val, 0.2f, beta);

    // Requantize: 将结果重新量化到输出的表示
    output_data = result / out_attrs.scale + out_attrs.zero_point;

    // 确保输出在有效范围内
    int final_output;
    if(is_signed) {
      final_output = static_cast<int8_t>(std::round(output_data)); 
      final_output = std::clamp(final_output, -128, 127); // 使用std::clamp确保范围
    } else {
      final_output = static_cast<uint8_t>(std::round(output_data)); // 对于无符号情况
      final_output = std::clamp(final_output, 0, 255); // 使用std::clamp确保范围
    }

    lut[i] = final_output;
    LOGS_DEFAULT(INFO)<<"Quantized HardSigmoid LUT value:"<<lut[i];
  }

  return lut;
}

std::vector<int32_t> GenerateQuantizedHardSwishLUT(phydnn_quant_param& out_attrs, 
                                                   phydnn_quant_param& in_attrs, 
                                                   size_t lut_size,
                                                   bool is_signed) {                                                
  std::vector<int32_t> lut(lut_size); // 假设输入是8位整数，所以大小为256

  for (int i = 0; i < lut_size; ++i) {
    int index = is_signed ? i - 128 : i;

    // Dequantize: 根据输入的scale和zero_point还原原始值
    float val = (index - in_attrs.zero_point) * in_attrs.scale;
    float output_data = hardswish(val);
    
    // Requantize: 将结果重新量化到输出的表示
    output_data = output_data / out_attrs.scale + out_attrs.zero_point;

    // 确保输出在有效范围内
    int final_output;
    if(is_signed) {
      final_output = static_cast<int8_t>(std::round(output_data)); 
      final_output = std::clamp(final_output, -128, 127); // 使用std::clamp确保范围
    } else {
      final_output = static_cast<uint8_t>(std::round(output_data)); // 对于无符号情况
      final_output = std::clamp(final_output, 0, 255); // 使用std::clamp确保范围
    }

    lut[i] = final_output;
    LOGS_DEFAULT(INFO)<<"Quantized HardSwish LUT value:"<<lut[i];
  }

  return lut;
}

// 生成Leakyrelu查找表
std::vector<int32_t> GenerateQuantizedLeakyreluLUT(phydnn_quant_param& out_attrs, 
                                                  phydnn_quant_param& in_attrs, 
                                                  size_t lut_size,
                                                  float alpha_value,
                                                  bool is_signed) {                                                
  std::vector<int32_t> lut(lut_size); // 假设输入是8位整数，所以大小为256

  for (int i = 0; i < lut_size; ++i) {
    int index = is_signed ? i - 128 : i;

    // Dequantize: 根据输入的scale和zero_point还原原始值
    float val = (index - in_attrs.zero_point) * in_attrs.scale;
    float output_data;
    if (val < 0) {
      // 计算负值情况下的输出
      float unclamped_output = val * alpha_value;
      val = unclamped_output;
    }

    // Requantize: 将结果重新量化到输出的表示
    output_data = val / out_attrs.scale + out_attrs.zero_point;

    // 确保输出在有效范围内
    int final_output;
    if(is_signed) {
      final_output = static_cast<int8_t>(std::round(output_data)); 
      final_output = std::clamp(final_output, -128, 127); // 使用std::clamp确保范围
    } else {
      final_output = static_cast<uint8_t>(std::round(output_data)); // 对于无符号情况
      final_output = std::clamp(final_output, 0, 255); // 使用std::clamp确保范围
    }

    lut[i] = final_output;
    LOGS_DEFAULT(INFO)<<"Quantized Leakyrelu LUT value:"<<lut[i];
  }

  return lut;
}

std::vector<int32_t> GenerateQuantizedSigmoidLUT(phydnn_quant_param& out_attrs, 
                                                phydnn_quant_param& in_attrs, 
                                                size_t lut_size,
                                                bool is_signed) {
  std::vector<int32_t> lut(lut_size); // 假设输入是8位整数，所以大小为256
  const float cutoff_upper = 16.619047164916992188f;
  const float cutoff_lower = -9.0f;

  for (int i = 0; i < lut_size; ++i) {
    int index = is_signed ? i - 128 : i;

    // Dequantize: 根据输入的scale和zero_point还原原始值
    float val = (index - in_attrs.zero_point) * in_attrs.scale;
    float output_data;
    if (val > cutoff_upper) {
      output_data = 1.0f;
    } else if (val < cutoff_lower) {
      output_data = std::exp(val);
    } else {
      output_data = 1.0f / (1.0f + std::exp(-val));
    }

    // Requantize: 将结果重新量化到输出的表示
    output_data = output_data / out_attrs.scale + out_attrs.zero_point;

    // 确保输出在有效范围内
    int final_output;
    if(is_signed) {
      final_output = static_cast<int8_t>(std::round(output_data)); 
      final_output = std::clamp(final_output, -128, 127); // 使用std::clamp确保范围
    } else {
      final_output = static_cast<uint8_t>(std::round(output_data)); // 对于无符号情况
      final_output = std::clamp(final_output, 0, 255); // 使用std::clamp确保范围
    }

    lut[i] = final_output;
    LOGS_DEFAULT(INFO)<<"Quantized Sigmoid LUT value:"<<lut[i];
  }

  return lut;
}

std::vector<int32_t> GenerateQuantizedTanhLUT(phydnn_quant_param& out_attrs, 
                                              phydnn_quant_param& in_attrs, 
                                              size_t lut_size,
                                              bool is_signed) {
  std::vector<int32_t> lut(lut_size); // 假设输入是8位整数，所以大小为256

  for (int i = 0; i < lut_size; ++i) {
    int index = is_signed ? i - 128 : i;

    // Dequantize: 根据输入的scale和zero_point还原原始值
    float val = (index - in_attrs.zero_point) * in_attrs.scale;

    // Apply Tanh
    float output_data = std::tanh(val);

    // Requantize: 将结果重新量化到输出的表示
    output_data = output_data / out_attrs.scale + out_attrs.zero_point;

    // 确保输出在有效范围内
    int final_output;
    if(is_signed) {
      final_output = static_cast<int8_t>(std::round(output_data)); 
      final_output = std::clamp(final_output, -128, 127); // 使用std::clamp确保范围
    } else {
      final_output = static_cast<uint8_t>(std::round(output_data)); // 对于无符号情况
      final_output = std::clamp(final_output, 0, 255); // 使用std::clamp确保范围
    }

    lut[i] = final_output;
    LOGS_DEFAULT(INFO)<<"Quantized Tanh LUT value:"<<lut[i];
  }

  return lut;
}

std::vector<int32_t> GenerateQuantizedExpLUT(phydnn_quant_param& out_attrs, 
                                              phydnn_quant_param& in_attrs, 
                                              size_t lut_size,
                                              bool is_signed) {
  std::vector<int32_t> lut(lut_size); // 假设输入是8位整数，所以大小为256

  for (int i = 0; i < lut_size; ++i) {
    int index = is_signed ? i - 128 : i;

    // Dequantize: 根据输入的scale和zero_point还原原始值
    float val = (index - in_attrs.zero_point) * in_attrs.scale;

    // Apply Exp
    float output_data = std::exp(val);

    // Requantize: 将结果重新量化到输出的表示
    output_data = output_data / out_attrs.scale + out_attrs.zero_point;

    // 确保输出在有效范围内
    int final_output;
    if(is_signed) {
      final_output = static_cast<int8_t>(std::round(output_data)); 
      final_output = std::clamp(final_output, -128, 127); // 使用std::clamp确保范围
    } else {
      final_output = static_cast<uint8_t>(std::round(output_data)); // 对于无符号情况
      final_output = std::clamp(final_output, 0, 255); // 使用std::clamp确保范围
    }

    lut[i] = final_output;
    LOGS_DEFAULT(INFO)<<"Quantized Exp LUT value:"<<lut[i];
  }

  return lut;
}


}  // namespace phytium_npu
}  // namespace onnxruntime
