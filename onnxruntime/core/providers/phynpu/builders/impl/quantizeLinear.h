 
//Data: 2026 -02-18

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class QuantizeLinearOpBuilder : public BaseOpBuilder {
 public:

  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating DequantizeLinear Op.";
    
    auto input_tensor   = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    size_t input_size = 1;
    for(unsigned int i=0; i<input_desc.dimensions; i++){
      input_size *= input_desc.size[i];
    }
    LOGS_DEFAULT(INFO)<<"QuantizeLinearOpBuilder input dim:"<<input_desc.dimensions<<", type:"<<input_desc.type<<", size:"
                      <<input_size<<", scale:"<<input_desc.quant_param.scale<<", zp:"<<input_desc.quant_param.zero_point;

    NodeAttrHelper helper(node_unit.GetNode());
    auto axis = helper.Get("axis", 0);
    if(axis < 0) {
      axis += input_desc.dimensions;
    }

  // get scale and zero_point
    const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
    float x_scale = 1.0;
    const auto& raw_scale_t = *initializers.at(node_unit.GetNode().InputDefs()[1]->Name());
    Initializer unpacked1_tensor(raw_scale_t);
    auto raw_scale = unpacked1_tensor.DataAsSpan<float>();
    auto scale_s = raw_scale[0];
    x_scale      = 1.0 / scale_s;

    int32_t y_zero_point  = 0;
    const auto& raw_zp_t = *initializers.at(node_unit.GetNode().InputDefs()[2]->Name());
    Initializer unpacked2_tensor(raw_zp_t);
    auto raw_zp  = unpacked2_tensor.DataAsSpan<int8_t>();
    y_zero_point = static_cast<int32_t>(raw_zp[0]);
    LOGS_DEFAULT(INFO)<<"QuantizeLinear get x_scale:"<<x_scale<<", y_zero_point:"<<y_zero_point;

  // set scale_tensor
    phydnn_tensor scale_tensor, zp_tensor;
    std::vector<float> s_data;
    std::vector<int> z_data;
    for(int i=0; i<input_size; i++) {
      s_data.push_back(x_scale);
      z_data.push_back(y_zero_point);
    }

    input_desc.type = PHYDNN_TYPE_F32;
    scale_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &input_desc, s_data.data(), &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
    }

  // set zero_point_tensor
    input_desc.type = PHYDNN_TYPE_I32;
    zp_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &input_desc, z_data.data(), &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
    }

  // 根据量化公式计算量化值: input * (1/scale) + zero_point
    auto output_tensor = MMBinaryBuildOp(graph_ep, input_tensor, scale_tensor, PHYDNN_OPERATION_MUL, node_unit); 

    input_desc.type = PHYDNN_TYPE_I32;
    output_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                    output_tensor,
                                    input_desc.type,
                                    nullptr,
                                    &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
      return output_tensors;
    } 

    output_tensor = phydnnNetworkBinaryOp(                                                 
            graph_ep->GetGraph(), output_tensor, zp_tensor, PHYDNN_OPERATION_ADD, &error_code);  
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkBinaryOp OPERATION_MAX, error_code:" << error_code; 
      return output_tensors;
    }  

  // cast to int8
    input_desc.type = PHYDNN_TYPE_Q_I8;
    input_desc.quant_param.scale = scale_s;
    input_desc.quant_param.zero_point = y_zero_point;
    output_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                    output_tensor,
                                    input_desc.type,
                                    &input_desc.quant_param,
                                    &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
      return output_tensors;
    } 

    //auto output_tensor = input_tensor;
    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output size:"<<output_desc.size[i]<<", type:"<<output_desc.type;
    }

    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }   
};  // QuantizeLinearOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
