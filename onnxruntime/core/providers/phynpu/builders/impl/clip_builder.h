 
//Data: 2026 -02-16

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {
class ClipOpBuilder : public BaseOpBuilder {
 protected:
  std::vector<phydnn_tensor> HandleBuildOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_quant_param in_quant,
            phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating clip Op.";
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    for(unsigned int i=0; i<input_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"ClipOpBuilder get result: "<<i<<" input_desc.size:"<<input_desc.size[i];
    }

    float min, max;
    GetphynpuClipMinMax(*graph_ep->GetGraphViewer(), node_unit.GetNode(), min, max);
    bool bMinFlag = min != std::numeric_limits<float>::lowest();
    bool bMaxFlag = max != std::numeric_limits<float>::max();

    auto output_tensor =
      phydnnNetworkReLUOp(graph_ep->GetGraph(), input_tensor, bMinFlag, min, bMaxFlag, max, 0, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkReLUOp!";
      return output_tensors;
    }

    
    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }
};

}  // namespace phytium_npu
}  // namespace onnxruntime
