 
 

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class ReluOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Relu Activation.";   

    auto input_tensor = *inputs[0];
    auto output_tensor = CreateReLULayer(graph_ep, input_tensor, true, 0.0, false, 0.0, false);
    return output_tensor;
  }
};

class Relu1OpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Relu1 Activation.";   

    auto input_tensor = *inputs[0];
    return CreateReLULayer(graph_ep, input_tensor, true, 0.0, false, 1.0, false);
  }
};

class Relu6OpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Relu6 Activation.";   

    auto input_tensor = *inputs[0];
    return CreateReLULayer(graph_ep, input_tensor, true, 0.0, false, 6.0, false);
  }
};

class LeakyReluOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_tensor output_tensor;
    phydnn_tensor_descriptor input_desc;
    phydnn_err_code error_code;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO) << "Creating LeakyRelu Activation input type:"<<input_desc.type;   

    const auto& node = node_unit.GetNode();
    NodeAttrHelper helper(node);
    auto alpha = helper.Get("alpha", 1.0f);

    if(input_desc.type >= PHYDNN_TYPE_Q_I8){
      LOGS_DEFAULT(INFO)<<"LeakyReluOpBuilder QuantizedLeakyrelu alpha:"<<alpha;
      // 定义输入范围和 LUT 大小
      const size_t lut_size = 257;

      bool is_quant = true;
      if (input_desc.type == PHYDNN_TYPE_Q_U8 || input_desc.type == PHYDNN_TYPE_QPA_U8) {
        is_quant = false;
      }

      // 生成 LUT
      std::vector<int32_t> Lut;
      Lut = GenerateQuantizedLeakyreluLUT(out_quant, in_quant, lut_size, alpha, is_quant);

      // 创建 LUT 查找操作
      output_tensor = phydnnNetworkInterpolatedLookupOp(
                                      graph_ep->GetGraph(),
                                      input_tensor,
                                      Lut.data(),
                                      lut_size,
                                      &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp!";
      }
      output_tensors.push_back(output_tensor);
      return output_tensors;
    } else {
      return CreateReLULayer(graph_ep, input_tensor, false, 0.0, false, 0.0, alpha);
    }
  }
};

class SignOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Sign Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_SIGN);
  }
};

class SigmoidOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {

    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_tensor output_tensor;
    phydnn_tensor_descriptor input_desc;
    phydnn_err_code error_code;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO) << "Creating Sigmoid Activation input type:"<<input_desc.type;   

    if(input_desc.type >= PHYDNN_TYPE_Q_I8){
      LOGS_DEFAULT(INFO)<<"SigmoidOpBuilder QuantizedSigmoid .";
      // 定义输入范围和 LUT 大小
      const size_t lut_size = 257;

      bool is_quant = true;
      if (input_desc.type == PHYDNN_TYPE_Q_U8 || input_desc.type == PHYDNN_TYPE_QPA_U8) {
        is_quant = false;
      }

      // 生成 LUT
      std::vector<int32_t> Lut;
      Lut = GenerateQuantizedSigmoidLUT(out_quant, in_quant, lut_size, is_quant);

      // 创建 LUT 查找操作
      output_tensor = phydnnNetworkInterpolatedLookupOp(
                                      graph_ep->GetGraph(),
                                      input_tensor,
                                      Lut.data(),
                                      lut_size,
                                      &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp!";
      }
      output_tensors.push_back(output_tensor);
      return output_tensors;
    } else {  
      return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_SIGMOID);
    }
  }
};

class TanhOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_tensor output_tensor;
    phydnn_tensor_descriptor input_desc;
    phydnn_err_code error_code;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO) << "Creating Tanh Activation input type:"<<input_desc.type;   

    if(input_desc.type >= PHYDNN_TYPE_Q_I8){
      LOGS_DEFAULT(INFO)<<"TanhOpBuilder QuantizedTanh .";
      // 定义输入范围和 LUT 大小
      const size_t lut_size = 257;

      bool is_quant = true;
      if (input_desc.type == PHYDNN_TYPE_Q_U8 || input_desc.type == PHYDNN_TYPE_QPA_U8) {
        is_quant = false;
      }

      // 生成 LUT
      std::vector<int32_t> Lut;
      Lut = GenerateQuantizedTanhLUT(out_quant, in_quant, lut_size, is_quant);

      // 创建 LUT 查找操作
      output_tensor = phydnnNetworkInterpolatedLookupOp(
                                      graph_ep->GetGraph(),
                                      input_tensor,
                                      Lut.data(),
                                      lut_size,
                                      &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp!";
      }
      output_tensors.push_back(output_tensor);
      return output_tensors;
    } else {  
      return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_TANH);
    }
  }
};

class NegateOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Negate Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_NEGATE);
  }
};

class AbsOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Abs Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_ABS);
  }
};

class NotOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Not Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_NOT);
  }
};

class LogOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Log Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_LOG);
  }
};

class SqrtOpBuilder : public BaseOpBuilder {
 public: 
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,                              
                      const Node* node) const override {                                         
    for (auto input : node->InputDefs()) {                                                      
      if (*input->Type() == "tensor(int8)" || *input->Type() == "tensor(uint8)") {                                                  
        LOGS_DEFAULT(INFO) << "Int8/UInt8 quant type is not supported as PHYDNN_OPERATION_SQRT ops."; 
        return false;                                                                           
      }                                                                                         
    }                                                                                           
    return true;                                                                                
  }  

  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Sqrt Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_SQRT);
  }
};

class ExpOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {

    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_tensor output_tensor;
    phydnn_tensor_descriptor input_desc;
    phydnn_err_code error_code;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO) << "Creating Exp Activation input type:"<<input_desc.type;  
    
    if(input_desc.type >= PHYDNN_TYPE_Q_I8){
      LOGS_DEFAULT(INFO)<<"ExpOpBuilder QuantizedExp .";
      // 定义输入范围和 LUT 大小
      const size_t lut_size = 257;

      bool is_quant = true;
      if (input_desc.type == PHYDNN_TYPE_Q_U8 || input_desc.type == PHYDNN_TYPE_QPA_U8) {
        is_quant = false;
      }

      // 生成 LUT
      std::vector<int32_t> Lut;
      Lut = GenerateQuantizedExpLUT(out_quant, in_quant, lut_size, is_quant);

      // 创建 LUT 查找操作
      output_tensor = phydnnNetworkInterpolatedLookupOp(
                                      graph_ep->GetGraph(),
                                      input_tensor,
                                      Lut.data(),
                                      lut_size,
                                      &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp!";
      }
      output_tensors.push_back(output_tensor);
      return output_tensors;
    } else {  
      return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_EXP);
    }
  }
};

class FloorOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Floor Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_FLOOR);
  }
};

class CeilOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Ceil Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_CEIL);
  }
};

class SinOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Sin Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_SIN);
  }
};

class RsqrtOpBuilder : public BaseOpBuilder {
 public: 
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Rsqrt Activation.";   

    auto input_tensor = *inputs[0];
    return CreateConvertUnaryActivations(graph_ep, input_tensor, PHYDNN_OPERATION_RSQRT);
  }
};

}  // namespace phytium_npu
}  // namespace onnxruntime