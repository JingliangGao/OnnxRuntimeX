 
//Data: 2026 -02-16

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class SpaceToDepthOpBuilder : public BaseOpBuilder {
 public:
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO) <<"SpaceToDepthOpBuilder input.type"<<input_desc.type<<", dims:"<<input_desc.dimensions; 

    NodeAttrHelper helper(node_unit);
    int blocksize = helper.Get("blocksize", 1);  
  
    auto output_tensor = phydnnNetworkSpaceToDepthOp(graph_ep->GetGraph(), 
                                                    input_tensor, 
                                                    blocksize, 
                                                    &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkSpaceToDepthOp!";
      return output_tensors;
    } 

    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }

    output_tensors.push_back(output_tensor);
    return output_tensors; 
  }   
};  // SpaceToDepthOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
