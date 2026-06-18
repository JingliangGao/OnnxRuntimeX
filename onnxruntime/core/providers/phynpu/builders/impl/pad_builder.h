 
 

#pragma once
#include <memory>
#include <vector>
#include <utility>
#include "core/providers/shared/utils/utils.h"
#include "core/providers/phynpu/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace phytium_npu {

class PadOpBuilder : public BaseOpBuilder {
 public: 

  std::vector<phydnn_tensor> HandleBuildOp(
    phytium_npu::GraphPhynpuEP* graph_ep,
    std::vector<std::shared_ptr<phydnn_tensor>>& inputs,
    const NodeUnit& node_unit,
    phydnn_quant_param in_quant,
    phydnn_quant_param out_quant) override {
    phydnn_tensor_descriptor input_desc;
    phydnn_err_code error_code;
    auto input_tensor = *inputs[0];
    std::vector<phydnn_tensor> output_tensors;
    phydnnGetTensorDescriptor(input_tensor, &input_desc);
    LOGS_DEFAULT(INFO)<<"PadOpBuilder input_desc.type:"<<input_desc.type;

    std::vector<int> pads;
    std::vector<int32_t> axes;
    std::vector<unsigned int> paddings; 
    float pad_value = 0.0;

    NodeAttrHelper helper(node_unit.GetNode());
    const auto mode = helper.Get("mode", "constant");
    if(node_unit.SinceVersion() < 11) {
      const auto raw_pads  = helper.Get("pads", std::vector<int>());
      const auto raw_value = helper.Get("value", 0.0f);
      pads.assign(raw_pads.begin(), raw_pads.end());
      pad_value = raw_value;
    } else {
      // get pads
      if(inputs.size() > 1) {
        const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
        const auto& pads_tensor = *initializers.at(node_unit.GetNode().InputDefs()[1]->Name());
        Initializer unpacked_tensor(pads_tensor);
        auto raw_pad = unpacked_tensor.DataAsSpan<int64_t>();
        const auto len = SafeInt<size_t>(pads_tensor.dims()[0]);
        pads.reserve(len);
        pads.insert(pads.end(), raw_pad.begin(), raw_pad.end());
        
        // get constant_value 
        if(inputs.size() > 2) {
          if(node_unit.GetNode().InputDefs()[2]->Name() == "") {
            pad_value = 0.0;
          } else {
            const auto& value_tensor = *initializers.at(node_unit.GetNode().InputDefs()[2]->Name());
            Initializer unpacked_tensor(value_tensor);
            pad_value = unpacked_tensor.DataAsSpan<float>()[0];
          }
        }
      }

      // get axes
      if(inputs.size() > 3) {
        const auto& initializers(graph_ep->GetGraphViewer()->GetAllInitializedTensors());
        const auto& axes_tensor = *initializers.at(node_unit.GetNode().InputDefs()[3]->Name());
        Initializer unpacked_tensor(axes_tensor);
        const ONNX_NAMESPACE::TypeProto* type_proto = node_unit.GetNode().InputDefs()[3]->TypeAsProto();
        auto data_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type_proto->tensor_type().elem_type());

        std::vector<int32_t> axes_in;
        if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
          auto raw_axes = unpacked_tensor.DataAsSpan<int64_t>();
          for (uint32_t i = 0; i < raw_axes.size(); i++) {
            int32_t data = SafeInt<int32_t>(raw_axes[i]);
            axes_in.push_back(data);
          }
        } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
          auto raw_axes = unpacked_tensor.DataAsSpan<int32_t>();
          for (uint32_t i = 0; i < raw_axes.size(); i++) {
            int32_t data = SafeInt<int32_t>(raw_axes[i]);
            axes_in.push_back(data);
          }
        }
        for (int i = 0; i < axes_in.size(); i++) {
          axes[i] = axes_in[i] < 0 ? axes_in[i] + input_desc.dimensions : axes_in[i];
        }
      }
    }

    for(unsigned int i=0; i<pads.size(); i++) {
      paddings.push_back(pads[i]);
    }

    // set pad_mode
    phydnn_pad_mode pad_mode = PHYDNN_PAD_CONSTANT;
    if (mode == "reflect") {
      pad_mode = PHYDNN_PAD_REFLECT;
    } else if (mode == "constant") {
      pad_mode = PHYDNN_PAD_CONSTANT;
    } else {
      pad_mode = PHYDNN_PAD_SYMMETRIC;
    }
    
    // set pad_value
    std::vector<unsigned int> start_padding;
    std::vector<unsigned int> end_padding;
    LOGS_DEFAULT(INFO)<<"PadOpBuilder pad_mode:"<<pad_mode;
    if(axes.size() > 0) {
      for (int i = 0; i < axes.size(); i++) {
        int index = axes[i];
        start_padding[index] = paddings[i];
        end_padding[index]   = paddings[i + paddings.size() / 2];
      }
    } else {
      start_padding.assign(paddings.begin(), paddings.begin() + paddings.size() / 2);
      end_padding.assign(paddings.begin() + paddings.size() / 2, paddings.end());
    }
   
    auto output_tensor = phydnnNetworkPadOp_v2(graph_ep->GetGraph(), 
                                              input_tensor,
                                              start_padding.data(), 
                                              end_padding.data(),
                                              pad_value, 
                                              pad_mode,
                                              &error_code);
    if(error_code != PHYDNN_SUCCESS) {
      LOGS_DEFAULT(ERROR) << "Failed to call phydnnNetworkPadOp_v2, error_code:" << error_code;
      return output_tensors;
    } 

    phydnn_tensor_descriptor output_desc;
    phydnnGetTensorDescriptor(output_tensor, &output_desc);
    for(unsigned int i=0; i<output_desc.dimensions; i++){
      LOGS_DEFAULT(INFO)<<"PadOpBuilder get result: "<<i<<" output_desc.size:"<<output_desc.size[i];
    }
    
    output_tensors.push_back(output_tensor);
    return output_tensors;   
  }
};

}  // namespace phytium_npu
}  // namespace onnxruntime
