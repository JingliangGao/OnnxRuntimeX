 
//Data: 2026 -02-18

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class SumOpBuilder : public BaseOpBuilder {
 public:
  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
      
    LOGS_DEFAULT(INFO) << "Creating Sum Op.";
    phydnn_err_code error_code;
    auto input_count = inputs.size();
    std::vector<phydnn_tensor> output_tensors;
    std::vector<phydnn_tensor> input_tensors;
    input_tensors.clear();
    for(int i=0; i<input_count; i++) {
      input_tensors.push_back(*inputs[i]);
    }

    auto output_tensor = input_tensors[0];
    for(int i=1; i<input_count; i++) {
      auto data = input_tensors[i];
      output_tensor = phydnnNetworkBinaryOp(graph_ep->GetGraph(), 
                                          output_tensor, 
                                          data, 
                                          PHYDNN_OPERATION_ADD, 
                                          &error_code);
      if(error_code != PHYDNN_SUCCESS) {
        LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkBinaryOp!";
        return output_tensors;
      }  
    }

    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions;i++){
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }

    output_tensors.push_back(output_tensor);
    return output_tensors;  
  }   
};  // SumOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
