 
//Data: 2026 -02-18

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class GatherOpBuilder : public BaseOpBuilder {
 public:
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Gather Op.";
    
    auto input_tensor   = *inputs[0];
    auto indices_tensor = *inputs[1];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc, indices_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    phydnnGetTensorDescriptor(indices_tensor, &indices_desc);
    LOGS_DEFAULT(INFO)<<"GatherOpBuilder input_desc.dim:"<<input_desc.dimensions<<" indices_desc.dim:"<<indices_desc.dimensions;

    NodeAttrHelper helper(node_unit.GetNode());
    auto axis = helper.Get("axis", 0);
    if(axis < 0) {
      axis += input_desc.dimensions;
    }
/*
    if(input_desc.type != indices_desc.type) {
      LOGS_DEFAULT(INFO)<<"BinaryOp input0 type:"<<input_desc.type<<", input1 type:"<<indices_desc.type;
      indices_tensor = phydnnNetworkCastOp(graph_ep->GetGraph(),
                                  indices_tensor,
                                  input_desc.type,
                                  &input_desc.quant_param,
                                  &error_code);
      if(error_code != PHYDNN_SUCCESS) {                                                          
        LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkCastOp, error_code:" << error_code; 
        return output_tensors;                                                                             
      }
    }
*/
    auto output_tensor = phydnnNetworkGatherOp(graph_ep->GetGraph(), 
                                              input_tensor, 
                                              indices_tensor, 
                                              axis, 
                                              &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkGatherOp!";
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
};  // GatherOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
