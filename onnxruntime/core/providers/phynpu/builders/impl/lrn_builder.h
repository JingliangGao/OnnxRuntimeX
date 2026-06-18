 
//Data: 2026 -02-18

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class LRNOpBuilder : public BaseOpBuilder {
 public:
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating LRN Op.";
    
    auto input_tensor  = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO)<<"LRNOpBuilder input_desc.dim:"<<input_desc.dimensions;

    NodeAttrHelper helper(node_unit.GetNode());
    const auto alpha = helper.Get("alpha", 0.0001f);
    const auto beta = helper.Get("beta", 0.75f);
    const auto bias = helper.Get("bias", 1.0f);  // k
    const auto size = helper.Get("size", 1);     // localSize
  
    auto output_tensor = phydnnNetworkLrnOp(graph_ep->GetGraph(), 
                                            input_tensor, 
                                            PHYDNN_LRN_ACROSS,
                                            size, bias, 
                                            alpha, beta,
                                            &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkLrnOp!";
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
};  // LRNOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
