 
//Data: 2026 -02-26

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {
class SplitOpBuilder : public BaseOpBuilder {
 public:
  std::vector<phydnn_tensor> HandleBuildOp(
            phytium_npu::GraphPhynpuEP* graph_ep,
            std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
            const NodeUnit& node_unit,
            phydnn_quant_param in_quant,
            phydnn_quant_param out_quant) override {
    LOGS_DEFAULT(INFO) << "Creating Cast Op.";
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnn_err_code error_code;
    phydnn_tensor_descriptor input_desc;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    for(unsigned int i=0; i<input_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"SplitOpBuilder get result: "<<i<<" input_desc.size:"<<input_desc.size[i];
    }

    NodeAttrHelper helper(node_unit.GetNode());
    int32_t axis = helper.Get("axis", 0);
    int32_t split_num = 1;
    std::vector<int32_t> split_data;

    if(node_unit.SinceVersion() >= 18 && helper.HasAttr("axes")) {
      int64_t num_outputs = helper.GetInt64("num_outputs").value();
      split_num = SafeInt<int32_t>(num_outputs);
    } else {
      split_num = SafeInt<int32_t>(node_unit.Outputs().size());
    }
    LOGS_DEFAULT(INFO)<<"get split axis:"<<axis<<", split_num:"<<split_num;

    if(node_unit.SinceVersion() <= 11) {
      split_data = helper.Get("split", std::vector<int32_t>{});
    } else {
      if(inputs.size() > 1) {
        const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
        const auto& split_tensor = *initializers.at(node_unit.GetNode().InputDefs()[1]->Name());
        Initializer unpacked_tensor(split_tensor);
        auto raw_split = unpacked_tensor.DataAsSpan<int64_t>();
        for (uint32_t i = 0; i < raw_split.size(); i++) {
          int32_t data = SafeInt<int32_t>(raw_split[i]);
          split_data.push_back(data);
        }
      }
    }

    for(int i=0; i<split_data.size(); i++) {
      LOGS_DEFAULT(INFO)<<"get split data: "<<i<<" split_data:"<<split_data[i];
    }

    std::vector<size_t> start(input_desc.dimensions, 0);
    std::vector<size_t> end(input_desc.dimensions, 0);
    std::vector<size_t> stride(input_desc.dimensions, 1);
    //split dimensions
    int split_axis = axis;
    if(split_axis < 0) {
      split_axis += input_desc.dimensions;
    }

    //1-split int, more-split list
    if(split_num == 1) {
      LOGS_DEFAULT(INFO)<<"SplitOpBuilder one split num:"<<split_num<<" split_axis:"<<split_axis;
      int num_outputs = split_data[0];
      int part_size   = input_desc.size[split_axis] / num_outputs;
      for(int i=0; i<num_outputs; i++) {
          //the last scale
          if(i == num_outputs-1 ) {
              start[split_axis] = part_size * i;
              end[split_axis]   = input_desc.size[split_axis] -1;
          } else {
              start[split_axis] = part_size * i;
              end[split_axis]   = part_size * (i+1) -1;
          }          
          
          phydnn_tensor tensor = phydnnNetworkSubTensor(graph_ep->GetGraph(),
                                                      input_tensor,
                                                      start.data(),
                                                      end.data(),
                                                      stride.data(),
                                                      &error_code);
          if(error_code != PHYDNN_SUCCESS) {
            LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkSubTensor!";
            return output_tensors;
          }                                
          output_tensors.push_back(tensor);
      }  
    } else {
      LOGS_DEFAULT(INFO)<<"SplitOpBuilder multi split_num:"<<split_num<<" split_axis:"<<split_axis;
      for(int i=0; i<input_desc.dimensions; i++) {
	      start[i] = 0;
        end[i] = input_desc.size[i]-1;
        stride[i] = 1;
      }

      start[split_axis] = 0;
      end[split_axis]   = 0;
      int itmp          = 0;
      for(int k=0; k<split_num; k++) {
          start[split_axis] = itmp;
          itmp             += split_data[k];
	        end[split_axis]   = (itmp-1);
          LOGS_DEFAULT(INFO)<<"SplitOpBuilder split_num:"<<k<<" itmp:"<<itmp<<" start:"<<start[split_axis]<<" end:"<<end[split_axis];
          if(start[split_axis] >= end[split_axis]) {
            LOGS_DEFAULT(INFO)<<"subTensor cout:"<<k<<"start:"<<start[split_axis]<<"end:"<<end[split_axis];
            continue;
          }
          phydnn_tensor tensor = phydnnNetworkSubTensor(graph_ep->GetGraph(),
                                                      input_tensor,
                                                      start.data(),
                                                      end.data(),
                                                      stride.data(),
                                                      &error_code);
          if(error_code != PHYDNN_SUCCESS) {
            LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkSubTensor!";
            return output_tensors;
          } 
	        output_tensors.push_back(tensor);
      }

      LOGS_DEFAULT(INFO)<<"CreateConvertSplit input size:"<<input_desc.size[split_axis]<<"itmp:"<<itmp;
      //the last scale
      if((input_desc.size[split_axis] - itmp) > 0) {
          start[split_axis] = itmp;
          end[split_axis]   = input_desc.size[split_axis] -1;
          phydnn_tensor tensor = phydnnNetworkSubTensor(graph_ep->GetGraph(),
                                                      input_tensor,
                                                      start.data(),
                                                      end.data(),
                                                      stride.data(),
                                                      &error_code);
          if(error_code != PHYDNN_SUCCESS) {
            LOGS_DEFAULT(ERROR) <<"Failed to call phydnnNetworkSubTensor!";
            return output_tensors;
          } 
          output_tensors.push_back(tensor);
      }
    }

    LOGS_DEFAULT(INFO)<<"SplitOpBuilder output tensors size:"<<output_tensors.size();
    for(int i=0; i<output_tensors.size(); i++) {
      phydnn_tensor_descriptor output_desc;
      phydnnGetTensorDescriptor(output_tensors[i], &output_desc);
      for(int k=0; k<output_desc.dimensions; k++){
        LOGS_DEFAULT(INFO)<<"SplitOpBuilder num:"<<i<<" output_desc.size:"<<output_desc.size[k];
      }
    }

    return output_tensors;
  }
};  // SplitOpBuilder

}  // namespace phytium_npu
}  // namespace onnxruntime
