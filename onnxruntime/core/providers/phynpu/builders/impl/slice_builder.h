 
//Data: 2026 -02-16

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {
class SliceOpBuilder : public BaseOpBuilder {
 public:
  std::vector<phydnn_tensor> HandleBuildOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_quant_param in_quant,
            phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Slice Op.";
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO)<<"SliceOpBuilder input_desc.dim:"<<input_desc.dimensions<<", inputs.size():"<<inputs.size();

    std::vector<int> starts;
    std::vector<int> ends;
    std::vector<int> axes;  
    axes.clear();
    std::vector<int> steps;

    // Opset 9 only has 1 input. The starts, ends, axes values are attributes.
    if (node_unit.SinceVersion() < 10) {
      NodeAttrHelper node_helper(node_unit);
      auto raw_starts = node_helper.Get("starts", std::vector<int64_t>{0});
      auto raw_ends = node_helper.Get("ends", std::vector<int64_t>{0});
      auto raw_axes = node_helper.Get("axes", std::vector<int64_t>{0});
      for(int i=0; i<raw_starts.size(); i++) {
        starts.push_back(raw_starts[i]);
      }
      for(int i=0; i<raw_ends.size(); i++) {
        ends.push_back(raw_ends[i]);
      }
      for(int i=0; i<raw_axes.size(); i++) {
        axes.push_back(raw_axes[i]);
      }
    } else {
      starts = GetDataFromInputTensors(graph_ep, inputs, node_unit, 1);
      ends   = GetDataFromInputTensors(graph_ep, inputs, node_unit, 2);
      if(inputs.size() > 3) {
        axes = GetDataFromInputTensors(graph_ep, inputs, node_unit, 3);
      }
      if(inputs.size() > 4) {
        steps= GetDataFromInputTensors(graph_ep, inputs, node_unit, 4);
      }
    }

    LOGS_DEFAULT(INFO)<<"SliceOpBuilder get params.";
    // get real start, end, stride
    std::vector<size_t> real_starts(input_desc.dimensions, 0);
    std::vector<size_t> real_ends(input_desc.dimensions, 0);
    std::vector<size_t> real_step(input_desc.dimensions, 1);
    
    for(size_t i = 0; i < input_desc.dimensions; ++i) {
      if(axes.size() > 0 && std::find(axes.begin(), axes.end(), i) != axes.end()) {
        int index = std::find(axes.begin(), axes.end(), i) - axes.begin(); 
        LOGS_DEFAULT(INFO)<<"SliceOpBuilder get i:"<<i<<", index:"<<index;
        int dim_size = input_desc.size[i];
        int start = starts[index] < 0 ? (starts[index] + dim_size) : starts[index];
        if(start > dim_size){
          start = dim_size - 1;
        }
        LOGS_DEFAULT(INFO)<<"SliceOpBuilder get start:"<<start;

        int end = ends[index] < 0 ? (ends[index] + dim_size -1) : (ends[index]-1);
        if(end >= dim_size){
          end = dim_size-1;
        }
        LOGS_DEFAULT(INFO)<<"SliceOpBuilder get end:"<<end;

        int step = 1;
        if(steps.size() > index){
          int step = steps[index] < 0 ? 1 : steps[index];
        }
        LOGS_DEFAULT(INFO)<<"SliceOpBuilder get step:"<<step;
        start = std::max(start, 0);
        end = std::max(end, 0);
        end = std::min(end, dim_size-1);
        real_starts[i] = start;
        real_ends[i] = end;
        real_step[i] = step;
      } else {
        real_starts[i] = 0;
        real_ends[i] = input_desc.size[i]-1;
        real_step[i] = 1;
      }
    }  

    LOGS_DEFAULT(INFO)<<"SliceOpBuilder phydnnNetworkSubTensor.";
    auto output_tensor = phydnnNetworkSubTensor(graph_ep->GetGraph(),
                                                input_tensor,
                                                real_starts.data(),
                                                real_ends.data(),
                                                real_step.data(),
                                                &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkSubTensor!";
      return output_tensors;
    }

    output_tensors.push_back(output_tensor);
    return output_tensors;
  }
};

}  // namespace phytium_npu
}  // namespace onnxruntime
