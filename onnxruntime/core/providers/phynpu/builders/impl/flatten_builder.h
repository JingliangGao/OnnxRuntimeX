 
//Data: 2026 -01-23

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class FlattenOpBuilder : public BaseOpBuilder {
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
    phydnn_tensor_descriptor input_desc, output_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    phydnnGetTensorDescriptor(input_tensor, &output_desc);
    for(unsigned int i=0; i<input_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"FlattenOpBuilder get result: "<<i<<" input_desc.size:"<<input_desc.size[i];
    }

    auto rank = input_desc.dimensions;
    NodeAttrHelper helper(node_unit.GetNode());
    int64_t axis = helper.Get("axis", 1);
    ORT_ENFORCE(axis < -rank || axis > rank, "Flatten Op axis: ", axis,
              " is not in valid range [-", rank, ",", rank, "]");
    if(axis < 0) {
      axis += rank;
    }
    LOGS_DEFAULT(INFO)<<"FlattenOpBuilder param axis: "<<axis;

    // set output_desc dims only 2-th
    output_desc.dimensions = 2;
    for(int i=0; i<rank; i++) {
      output_desc.size[i] = 1;
    }  
    for (int64_t i = 0; i < axis; i++) {
      output_desc.size[0] *= input_desc.size[i];
    }
    for(int64_t i = axis; i < rank; i++) {
      output_desc.size[1] *= input_desc.size[i];
    }

    auto output_tensor =
        phydnnNetworkReshapeOp(graph_ep->GetGraph(), input_tensor, &output_desc, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReshapeOp!";
      return output_tensors;
    }    
    
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(unsigned int i=0; i<output_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"FlattenOpBuilder get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }

    
    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }   
};  // FlattenOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
