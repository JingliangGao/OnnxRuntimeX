 
//Data: 2026 -02-18

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class PReluOpBuilder : public BaseOpBuilder {
 public:
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating PRelu Op.";
    auto input_tensor = *inputs[0];
    auto slope_tensor = *inputs[1];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc, slope_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    phydnnGetTensorDescriptor(slope_tensor, &slope_desc);
    LOGS_DEFAULT(INFO)<<"PReluOpBuilder input dim:"<<input_desc.dimensions<<" slope dim:"<<slope_desc.dimensions
                      <<", input type:"<<input_desc.type<<", slope type:"<<slope_desc.type;

    if(slope_desc.dimensions < input_desc.dimensions) {
      for(int i = slope_desc.dimensions; i<input_desc.dimensions; i++){
        slope_tensor = phydnnNetworkBroadcastOp(graph_ep->GetGraph(), slope_tensor, i, input_desc.size[i], &error_code);
        LOGS_DEFAULT(ERROR)<< "Failed to call phydnnNetworkBroadcastOp!";
      }
    }

    // calculate min
    const float fscale = 0.0;
    phydnn_tensor_descriptor scale_desc = {1, PHYDNN_TYPE_F32, {1}, {{{1.0f, 0}}}};
    scale_desc.quant_param = input_desc.quant_param;
    phydnn_tensor zero = phydnnNetworkFixedInput(graph_ep->GetGraph(), &scale_desc, &fscale, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkFixedInput!";
      return output_tensors;
    }

    phydnn_tensor min = phydnnNetworkBinaryOp(graph_ep->GetGraph(), input_tensor, zero, PHYDNN_OPERATION_MIN, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkBinaryOp!";
      return output_tensors;
    }

    min = phydnnNetworkBinaryOp(graph_ep->GetGraph(), min, slope_tensor, PHYDNN_OPERATION_MUL, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkBinaryOp!";
      return output_tensors;
    }

    phydnn_tensor max = phydnnNetworkBinaryOp(graph_ep->GetGraph(), input_tensor, zero, PHYDNN_OPERATION_MAX, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkBinaryOp!";
      return output_tensors;
    }

    phydnn_tensor output_tensor = phydnnNetworkBinaryOp(graph_ep->GetGraph(), min, max, PHYDNN_OPERATION_ADD, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkBinaryOp!";
      return output_tensors;
    }
    
    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions;i++){
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }

    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }   
};  // PReluOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
