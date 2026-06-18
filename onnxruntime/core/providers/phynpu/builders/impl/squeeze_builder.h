 
//Data: 2026 -02-16

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {
class SqueezeOpBuilder : public BaseOpBuilder {
 protected:
  std::vector<phydnn_tensor> HandleBuildOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_quant_param in_quant,
            phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Squeeze Op.";
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc, input1_desc, dst_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    for(unsigned int i=0; i<input_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"SqueezeOpBuilder get result: "<<i<<" input_desc.size:"<<input_desc.size[i];
    }

    // Squeeze opset 13 use input as axes
    NodeAttrHelper helper(node_unit.GetNode());
    std::vector<int64_t> def_axes;
    if (node_unit.SinceVersion() > 12) {
      const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
      const auto& axes_tensor = *initializers.at(node_unit.GetNode().InputDefs()[1]->Name());
      Initializer unpacked_tensor(axes_tensor);
      auto raw_axes = unpacked_tensor.DataAsSpan<int64_t>();
      const auto size = SafeInt<size_t>(axes_tensor.dims()[0]);
      def_axes.reserve(size);
      def_axes.insert(def_axes.end(), raw_axes.begin(), raw_axes.end());
    } else {
      def_axes = helper.Get("axes", std::vector<int64_t>());
    }

    std::vector<int32_t> axes(def_axes.begin(), def_axes.end());
    std::unordered_set<int> axis_checker;
    std::vector<int> oshape;
    for(int i=0; i<axes.size(); i++) {
      int real_axes = 0;
      if(axes[i] < 0) {
        real_axes = axes[i] + input_desc.dimensions;
      } else {
        real_axes = axes[i];
      }

      if(real_axes >= input_desc.dimensions || real_axes < 0) {
        continue;
      } else {
        axis_checker.insert(real_axes);
      }
    }

    for(int i=0; i < input_desc.dimensions; i++) {
      if(axis_checker.find(i) == axis_checker.end()) {
        oshape.push_back(input_desc.size[i]);
      } else {
        if(input_desc.size[i] != 1){
          LOGS_DEFAULT(INFO)<< "The squeezed axis must have shape 1 Want to squeeze:"<<input_desc.size[i];
        }
      }
    }

    dst_desc.type = input_desc.type;
    dst_desc.dimensions = oshape.size();
    if(dst_desc.dimensions > PHYDNN_DESCRIPTOR_MAX_DIM){
      LOGS_DEFAULT(INFO)<< "CreateConvertSqueeze over dim limits:"<<dst_desc.dimensions;
      return output_tensors;
    }
    for(int i=0; i<oshape.size(); i++) {
      dst_desc.size[i] = oshape[i];
    }

    auto output_tensor =
         phydnnNetworkReshapeOp(graph_ep->GetGraph(), input_tensor, &dst_desc, &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call CreateConvertSqueeze!";
      return output_tensors;
    }

    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(int i=0; i<output_desc.dimensions; i++) {
      LOGS_DEFAULT(INFO)<<"get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }
    
    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }
};

}  // namespace phytium_npu
}  // namespace onnxruntime
