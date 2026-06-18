 
//Data: 2026 -02-16

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class TransposeOpBuilder : public BaseOpBuilder {
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
    LOGS_DEFAULT(INFO) <<"TransposeOpBuilder input_desc.type"<<input_desc.type; 

    NodeAttrHelper helper(node_unit.GetNode());
    std::vector<int64_t> perm = helper.Get("perm", std::vector<int64_t>());
    std::vector<int> order;
    if (perm.empty()) {
      int num_axes = input_desc.dimensions;
      for (int i = num_axes - 1; i > -1; --i)
        order.push_back(i);
    } else {
      for (uint32_t i = 0; i < perm.size(); ++i)
        order.push_back(perm[i]);
    }

    auto output_tensor = phydnnNetworkTransposeOp(graph_ep->GetGraph(),
                                                input_tensor,
                                                order.data(),
                                                &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkTransposeOp!";
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
};  // TransposeOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
