 
 

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class ConcatOpBuilder : public BaseOpBuilder {
 public: 
  bool IsOpSupported(const onnxruntime::GraphViewer& graph_viewer,
                     const Node* node) const override {
    NodeAttrHelper helper(*node);
    auto axis = helper.Get("axis", 0);
    auto input_defs = node->InputDefs();
    auto input_shape = GetTensorShape(*input_defs[0]);
    int32_t rank = input_shape.NumDimensions();
    if (axis >= rank || axis < -rank) {
      LOGS_DEFAULT(ERROR) << "Axis is invalid in Concat.";
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
    phydnn_err_code error_code;
    std::vector<phydnn_tensor> input_tensors;
    std::vector<phydnn_tensor> output_tensors;
    const auto input_count = inputs.size();

    // densenet121模型中出现concat只有一个输入的情况，此时直接返回输入tensor
    if(inputs.size() == 1) {
      LOGS_DEFAULT(INFO) << "ConcatOpBuilder only one inputs, so return the input tensor now.";
      auto input_tensor = *inputs[0];
      output_tensors.push_back(input_tensor);
      return output_tensors;   
    }

    for (int i=0; i<input_count; i++) {
      auto input_tensor = *inputs[i];
      input_tensors.push_back(input_tensor);
    }

    phydnn_tensor_descriptor input_desc[input_count];
    for(int i=0; i<input_count; i++){
      phydnnGetTensorDescriptor(input_tensors[i], &input_desc[i]);
      LOGS_DEFAULT(INFO)<<"Createconcat input dims:"<<input_desc[i].dimensions<<", type:"<<input_desc[i].type;
      for(int k=0; k<input_desc[i].dimensions; k++){
        LOGS_DEFAULT(INFO)<<"Createconcat input_desc K: "<<k<<" size:"<<input_desc[i].size[k];
      }
    }

    // get attrs
    NodeAttrHelper helper(node_unit.GetNode());
    auto axis = helper.Get("axis", 0);
    if(axis < 0) {
      axis += input_desc[0].dimensions;  
    }

    LOGS_DEFAULT(INFO) << "Concat Op by inputs cout:"<<input_count<<", input size:"<<input_tensors.size()<<", axis:"<<axis;   
    auto output_tensor = phydnnNetworkConcatOp(graph_ep->GetGraph(),
                                              input_tensors.data(),
                                              axis,
                                              input_count,
                                              &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkConcatOp, error_code:" << error_code;
      return output_tensors;
    } 

    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(unsigned int i=0; i<output_desc.dimensions; i++) {
       LOGS_DEFAULT(INFO)<<"ConcatOpBuilder get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }
    
    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }
};


}  // namespace phytium_npu
}  // namespace onnxruntime
