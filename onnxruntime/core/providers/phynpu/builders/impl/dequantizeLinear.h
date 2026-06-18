 
//Data: 2026 -02-18

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class DequantizeLinearOpBuilder : public BaseOpBuilder {
 public:
   bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    NodeAttrHelper helper(*node);
    const auto axis = helper.Get("axis", 1);
    if (axis != 1) {
      LOGS_DEFAULT(WARNING) << "DequantizeLinear Only can support the axis==1.";
      return false;
    }
    return true;
  }

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
    LOGS_DEFAULT(INFO)<<"DequantizeLinearOpBuilder input_desc.dim:"<<input_desc.dimensions<<", type:"<<input_desc.type<<", input_size:"<<input_size;

    NodeAttrHelper helper(node_unit.GetNode());
    auto axis = helper.Get("axis", 0);
    if(axis < 0) {
      axis += input_desc.dimensions;
    }

    if(input_desc.type >= 8) {
      input_tensor = phydnnNetworkCastOp(
                    graph_ep->GetGraph(), input_tensor, PHYDNN_TYPE_I32, nullptr, &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
        return output_tensors;
      } 
    }
  
  // get x_scale 
    float x_scale = 0.0;
    const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
    const auto& raw_scale_t = *initializers.at(node_unit.GetNode().InputDefs()[1]->Name());
    Initializer unpacked_tensor(raw_scale_t);
    auto raw_scale = unpacked_tensor.DataAsSpan<float>();
    x_scale = raw_scale[0];

  // get x_zero_point
    int32_t x_zero_point = 0;
    int8_t raw_zp = 0;
    if(inputs.size() > 2) {
      const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
      const auto& zero_point_tensor = *initializers.at(node_unit.GetNode().InputDefs()[2]->Name());
      Initializer unpacked_tensor(zero_point_tensor);
      auto raw_zero_point = unpacked_tensor.DataAsSpan<int8_t>();
      raw_zp = raw_zero_point[0];
      x_zero_point  = static_cast<int32_t>(raw_zp);
    }
    LOGS_DEFAULT(INFO)<<"get x_scale:"<<x_scale<<", x_zero_point:"<<x_zero_point<<", raw_zp:"<<raw_zp;

  // get zero_point and scale_data tensor
    std::vector<int32_t> zp_data;
    std::vector<float> scale_data;
    for(int i=0; i<input_size; i++) {
      zp_data.push_back(x_zero_point);
      scale_data.push_back(x_scale);
    }

    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    phydnn_tensor zp_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &input_desc, zp_data.data(), &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
      return output_tensors;
    }

    input_desc.type = PHYDNN_TYPE_F32;
    phydnn_tensor scale_tensor = phydnnNetworkFixedInput(graph_ep->GetGraph(), &input_desc, scale_data.data(), &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkFixedInput!";
      return output_tensors;
    }

  // y = (x − x_zero_point​) * x_scale​
    auto output_tensor = phydnnNetworkBinaryOp(                                                 
        graph_ep->GetGraph(), input_tensor, zp_tensor, PHYDNN_OPERATION_SUB, &error_code);  
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkBinaryOp SUB!";
      return output_tensors;
    }  

    // cast
    output_tensor = phydnnNetworkCastOp(
                    graph_ep->GetGraph(), output_tensor, PHYDNN_TYPE_F32, nullptr, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code;
      return output_tensors;
    } 

    output_tensor = phydnnNetworkBinaryOp(                                                 
        graph_ep->GetGraph(), output_tensor, scale_tensor, PHYDNN_OPERATION_MUL, &error_code);  
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkBinaryOp MUL!";
      return output_tensors;
    }  

    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output size:"<<output_desc.size[i]<<", type:"<<output_desc.type;
    }

    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }   
};  // DequantizeLinearOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
